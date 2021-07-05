import itertools
import logging
import math
import multiprocessing
import os
import pandas as pd
import re
import sys
import xml.etree.cElementTree as ET
import _pickle as pickle
sys.path.append('..')
from configures import Configs
from Corpus.preprocessor import Preprocessor
from tqdm import tqdm
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

preprocessor = Preprocessor()
NUM_CPU = None


class Corpus:
    """consider the commit version"""

    def __init__(self, corpus_dir, report_path=None, code_dir=None):
        self.report_path = report_path
        self.code_dir = code_dir
        self.corpus_dir = corpus_dir
        self.collected_codes_dir = f'{corpus_dir}/collected_codes'
        self.preprocessed_codes_dir = f'{corpus_dir}/preprocessed_codes'
        self.preprocessed_reports_dir = f'{corpus_dir}/preprocessed_reports'

        self.code_corpus_path = f'{self.preprocessed_codes_dir}/code_corpus.csv'
        self.code_camel_record_path = f'{self.preprocessed_codes_dir}/camel_record.pkl'
        self.commit2commit_code_paths_path = f'{self.preprocessed_codes_dir}/commit2commit_code_paths.pkl'

    def generate_code_corpus(self, len_snippet):
        assert self.report_path is not None and self.code_dir is not None
        corpus = CodeCorpus(self.corpus_dir, self.report_path, self.code_dir)
        corpus.generate_corpus(len_snippet)

    def generate_report_corpus(self):
        assert self.report_path is not None and self.code_dir is not None
        corpus = ReportCorpus(self.corpus_dir, self.report_path, self.code_dir)
        corpus.generate_corpus()

    def load_code_corpus(self):
        code_corpus = pd.read_csv(self.code_corpus_path)
        code_corpus.fillna('', inplace=True)
        return code_corpus

    def load_code_camel_word_record(self):
        with open(self.code_camel_record_path, 'rb') as f:
            record = pickle.load(f)
        return record

    def load_commit2commit_code_paths(self):
        """{commit: {path: valid_commit/path}}"""
        with open(self.commit2commit_code_paths_path, 'rb') as f:
            commit2commit_code_paths = pickle.load(f)
        return commit2commit_code_paths

    def load_commit2code_paths(self):
        commit2commit_code_paths = self.load_commit2commit_code_paths()
        commit2code_paths = dict([
            (commit, list(commit2commit_code_paths[commit].keys())) for commit in commit2commit_code_paths
        ])
        return commit2code_paths

    def load_collected_codes(self):
        codes_df = pd.read_csv(f'{self.collected_codes_dir}/collected_codes.csv')
        codes_df.fillna('', inplace=True)
        return codes_df

    def load_report_corpus(self, tag):
        assert tag in ('train', 'val', 'test', 'total')
        if tag == 'total':
            report_corpus = self._load_report_corpus('train')
            report_corpus = pd.concat([report_corpus, self._load_report_corpus('val')])
            report_corpus = pd.concat([report_corpus, self._load_report_corpus('test')])
        else:
            report_corpus = self._load_report_corpus(tag)
        return report_corpus

    def _load_report_corpus(self, tag):
        path = f'{self.preprocessed_reports_dir}/{tag}_report_corpus.csv'
        report_corpus = pd.read_csv(path)
        report_corpus.fillna('', inplace=True)
        return report_corpus

    def load_report_camel_word_record(self, tag):
        assert tag in ('train', 'val', 'test', 'total')
        if tag == 'total':
            camel_words = self._load_report_camel_word_record('train')
            camel_words.update(self._load_report_camel_word_record('val'))
            camel_words.update(self._load_report_camel_word_record('test'))
        else:
            camel_words = self._load_report_camel_word_record(tag)
        return camel_words

    def _load_report_camel_word_record(self, tag):
        path = f'{self.preprocessed_reports_dir}/{tag}_report_camel_record.pkl'
        with open(path, 'rb') as f:
            camel_record = pickle.load(f)
        return camel_record


class CodeCorpus(Corpus):
    def __init__(self, corpus_dir, report_path, code_dir):
        super().__init__(corpus_dir, report_path, code_dir)

    def generate_corpus(self, len_snippet):
        logging.info('load collected codes...')
        code_df = self.load_collected_codes()

        logging.info('preprocess codes...')
        code_contents = code_df['content'].tolist()
        code_tokens, camel_words_record = self.preprocess(code_contents)

        logging.info('generate code snippets...')
        snippets = self.split_codes(code_tokens, len_snippet)
        corpus = code_df
        corpus.columns = ['commit', 'path', 'snippets']
        corpus['snippets'] = snippets

        logging.info('save code_corpus and camel_word_record...')
        self.save_corpus(corpus, camel_words_record)

        # For training embedding
        with open(f'{self.preprocessed_codes_dir}/preprocessed_code_tokens.pkl', 'wb') as f:
            pickle.dump(code_tokens, f)

        logging.info('index code paths...')
        with open(f'{self.collected_codes_dir}/commit_order.txt', 'r') as f:
            commit_order = f.read().split('\n')
        with open(f'{self.collected_codes_dir}/commit2code_paths.pkl', 'rb') as f:
            commit2code_paths = pickle.load(f)
        commit2commit_code_paths = self.index_code_paths(corpus, commit_order, commit2code_paths)

        logging.info('save code path index...')
        self.save_code_path_index(commit2commit_code_paths)
        logging.info('done.')

    @staticmethod
    def preprocess(code_contents):
        camel_word_record = {}
        code_tokens = []
        code_contents = list(zip(code_contents, range(len(code_contents))))
        logging.info(f'multiprocessing: NUM_CPU = {NUM_CPU}')

        with multiprocessing.Pool(NUM_CPU) as p:
            results = list(tqdm(
                p.imap(preprocessor.preprocess_code_with_multiprocess, code_contents),
                total=len(code_contents),
                ncols=80,
                desc=f'preprocessing code'
            ))
        results = list(sorted(results, key=lambda x: x[2]))

        for each_tokens, each_record, _ in results:
            code_tokens.append(each_tokens)
            camel_word_record.update(each_record)
        return code_tokens, camel_word_record

    @staticmethod
    def split_codes(codes, len_snippet):
        code_snippets = []
        for tokens in codes:
            snippets = [' '.join(tokens[i * len_snippet: (i + 1) * len_snippet])
                        for i in range(math.ceil(len(tokens) / len_snippet))]
            code_snippets.append('\n'.join(snippets))
        return code_snippets

    def index_code_paths(self, code_corpus, commit_order, commit2code_paths):
        """
        index paths of all codes of every commit with valid paths.
        for example, if a code path at commit '123456' is 'java/a.java', and the code is saved at commit '000000',
        then the valid path is '000000/java/a.java'
        :return {commit: {path: valid_commit/path}}
        """
        with multiprocessing.Pool(NUM_CPU, initializer=self.initializer,
                                  initargs=(code_corpus, commit_order, commit2code_paths)) as p:
            results = list(tqdm(
                p.imap(self.get_each_commit_code_path_index, range(len(commit_order))),
                total=len(commit_order),
                ncols=80
            ))
        all_commit_code_path_index = dict(results)
        return all_commit_code_path_index

    @staticmethod
    def initializer(code_corpus, commit_order, commit2code_paths):
        global code_corpus_global
        global commit2code_paths_global
        global commit_order_global
        code_corpus_global = code_corpus
        commit2code_paths_global = commit2code_paths
        commit_order_global = commit_order

    @staticmethod
    def get_each_commit_code_path_index(commit_idx):
        current_commit = commit_order_global[commit_idx]
        sub_commit_order_global = commit_order_global[:commit_idx + 1]
        code_path_index = {}
        current_commit_all_code_paths = commit2code_paths_global[current_commit]
        codes_candidate = code_corpus_global[
            code_corpus_global.path.isin(current_commit_all_code_paths) &
            code_corpus_global.commit.isin(sub_commit_order_global)
            ]
        for commit in sub_commit_order_global[::-1]:
            codes_part = codes_candidate[codes_candidate.commit == commit]
            for c in codes_part.itertuples():
                if c.path not in code_path_index:
                    code_path_index[c.path] = f'{commit}/{c.path}'

        assert set(current_commit_all_code_paths) == set(code_path_index.keys())
        return current_commit, code_path_index

    def save_corpus(self, code_corpus, record):
        if not os.path.exists(os.path.dirname(self.code_corpus_path)):
            os.makedirs(os.path.dirname(self.code_corpus_path))
        code_corpus.to_csv(self.code_corpus_path, index=False)

        with open(self.code_camel_record_path, 'wb') as f:
            pickle.dump(record, f)

    def save_code_path_index(self, commit2commit_code_paths):
        with open(self.commit2commit_code_paths_path, 'wb') as f:
            pickle.dump(commit2commit_code_paths, f)


class ReportCorpus(Corpus):
    def __init__(self, corpus_dir, report_path, code_dir):
        super().__init__(corpus_dir, report_path, code_dir)
        self.commit2paths = None

    def generate_corpus(self):
        logging.info('load and split reports...')
        train_val_test_reports = self.load_and_split_reports()

        logging.info('load commit2code_paths...')
        self.commit2paths = self.load_commit2code_paths()

        for reports, tag in zip(train_val_test_reports, ('train', 'val', 'test')):
            logging.info(f'generate {tag} report corpus...')
            corpus, camel_words_record = self.preprocess(reports)
            print(f'number of {tag} reports: {len(corpus)}')

            logging.info(f'save report corpus and camel record...')
            self.save_corpus(corpus, camel_words_record, tag)

    def load_and_split_reports(self, split_ratio='8:1:1'):
        """
        sort reports by 'report_timestamp', then split.
        """
        root = ET.parse(self.report_path).getroot()
        reports = list(root.iter('table'))
        reports = list(sorted(reports, key=lambda x: int(x[5].text)))
        n_train = int(int(split_ratio.split(':')[0]) / 10 * len(reports))
        n_val = int(int(split_ratio.split(':')[1]) / 10 * len(reports))
        train_reports = reports[:n_train]
        val_reports = reports[n_train: n_train + n_val]
        test_reports = reports[n_train + n_val:]
        print(f'split_ratio = {split_ratio}')
        print(f'train reports: {len(train_reports)}')
        print(f'val   reports: {len(val_reports)}')
        print(f'test  reports: {len(test_reports)}')
        return train_reports, val_reports, test_reports

    def preprocess(self, reports):
        corpus = []
        camel_word_record = {}
        for r in tqdm(reports, ncols=80):
            commit = r[7].text
            buggy_file_paths = r[9].text.split('\n')
            buggy_file_paths = self.filter_buggy_path(buggy_file_paths, commit)
            if not buggy_file_paths:
                continue

            bug_id = r[1].text
            summary = r[2].text
            description = r[3].text if r[3].text else ''
            report_timestamp = int(r[5].text)
            commit_timestamp = int(r[8].text)

            summary_tokens, camel_info = preprocessor.preprocess_report(summary, tag='summary')
            keywords_summary = list(itertools.chain(*camel_info.values())) + list(camel_info.keys())
            keywords_summary = list(set(keywords_summary))
            camel_word_record.update(camel_info)

            description_tokens, camel_info = preprocessor.preprocess_report(description, tag='description')
            keywords_description = list(itertools.chain(*camel_info.values())) + list(camel_info.keys())
            keywords_description = list(set(keywords_description))
            camel_word_record.update(camel_info)

            code_in_trace = self.extract_code_in_trace(description)

            pr = [
                bug_id, commit, ' '.join(summary_tokens), ' '.join(description_tokens),
                ' '.join(keywords_summary), ' '.join(keywords_description),
                '\n'.join(buggy_file_paths), '\n'.join(code_in_trace), report_timestamp, commit_timestamp
            ]
            corpus.append(pr)
        return corpus, camel_word_record

    def filter_buggy_path(self, buggy_file_paths, commit):
        """filter out buggy files that are 'ADD'"""
        valid_paths = [path for path in buggy_file_paths if path in self.commit2paths[commit]]
        return valid_paths

    @staticmethod
    def extract_code_in_trace(description):
        f_name = re.findall(r'\((.+?\.java):\d+\)', description)
        for line in description.split('\n'):
            if '(unknown source)' in line.lower():
                idx = line.lower().index('(unknown source)')
                line_split = line[:idx].split('.')
                if len(line_split) > 1:
                    name = line_split[-2] + '.java'
                    f_name.append(name)
        return list(set(f_name))

    def save_corpus(self, corpus, camel_record, tag):
        path = f'{self.preprocessed_reports_dir}/{tag}_report_corpus.csv'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        columns = ['bug_id', 'commit', 'summary', 'description', 'keywords_summary', 'keywords_description',
                   'buggy_paths', 'trace', 'report_timestamp', 'commit_timestamp']
        df = pd.DataFrame(data=corpus, columns=columns)
        df.to_csv(path, index=False)

        path = f'{self.preprocessed_reports_dir}/{tag}_report_camel_record.pkl'
        with open(path, 'wb') as f:
            pickle.dump(camel_record, f)


def main(project_name):
    configs = Configs(project_name)
    print(f"corpus_dir: {configs.corpus_dir}")
    print(f'length of snippet: {configs.len_code_snippet}\n')

    global NUM_CPU
    NUM_CPU = min(configs.num_cpu, multiprocessing.cpu_count() - 1)

    corpus = Corpus(configs.corpus_dir, configs.report_path, configs.code_dir)
    corpus.generate_code_corpus(configs.len_code_snippet)
    corpus.generate_report_corpus()


if __name__ == '__main__':
    project_name = sys.argv[1]
    main(project_name)
