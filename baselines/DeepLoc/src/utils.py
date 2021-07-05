import os
import pandas as pd
import inflection
import _pickle as pickle
import xml.etree.cElementTree as ET
import re
from nltk.tokenize import sent_tokenize
from sklearn.externals import joblib
from string import punctuation
from assets import *
filter_words_set = stop_words.union(java_keywords)


def load_file(path):
    assert path[-4:] == '.pkl'
    try:
        with open(path, 'rb') as f:
            file = pickle.load(f)
    except FileNotFoundError:
        with open(f'{path[:-4]}.joblib', 'rb') as f:
            file = joblib.load(f)
    return file


def save_file(path, file):
    assert path[-4:] == '.pkl'
    try:
        with open(path, 'wb') as f:
            pickle.dump(file, f)
    except OverflowError:
        os.remove(path)
        with open(f'{path[:-4]}.joblib', 'wb') as f:
            joblib.dump(file, f)


def check_dir(path):
    basename = os.path.basename(path)
    if '.' not in basename:
        directory = path
    else:
        directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_report(report_path, tag, split_ratio='8:1:1'):
    """
    sort reports by 'report_timestamp', then split.
    """
    assert tag in ('split', 'total')
    root = ET.parse(report_path).getroot()
    reports = list(root.iter('table'))
    reports = list(sorted(reports, key=lambda x: int(x[5].text)))
    if tag == 'total':
        return reports
    else:
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


def load_report_corpus(tag, report_corpus_dir):
    assert tag in ('train', 'val', 'test', 'train_val', 'total')
    if tag == 'total':
        report_corpus = _load_report_corpus('train', report_corpus_dir)
        report_corpus = pd.concat([report_corpus, _load_report_corpus('val', report_corpus_dir)])
        report_corpus = pd.concat([report_corpus, _load_report_corpus('test', report_corpus_dir)])
    elif tag == 'train_val':
        report_corpus = _load_report_corpus('train', report_corpus_dir)
        report_corpus = pd.concat([report_corpus, _load_report_corpus('val', report_corpus_dir)])
    else:
        report_corpus = _load_report_corpus(tag, report_corpus_dir)
    return report_corpus


def _load_report_corpus(tag, report_corpus_dir):
    assert tag in ('train', 'val', 'test')
    path = f'{report_corpus_dir}/{tag}_report_corpus.csv'
    report_corpus = pd.read_csv(path)
    report_corpus.fillna('', inplace=True)
    return report_corpus


def load_code_corpus(code_corpus_dir):
    path = f'{code_corpus_dir}/preprocessed_code_tokens.pkl'
    code_corpus = load_file(path)
    return code_corpus


def tokenize(sentence):
    filter_sent = re.sub('[^a-zA-Z]', ' ', sentence)
    tokens = filter_sent.split()
    return tokens


def split_camelcase(tokens, retain_camelcase=True):
    """
    :param tokens: [str]
    :param retain_camelcase: if True, the corpus will retain camel words after splitting them.
    :return:
    """
    def split_by_punc(token):
        new_tokens = []
        split_toks = re.split(fr'[{punctuation}]+', token)
        if len(split_toks) > 1:
            return_tokens.remove(token)
            for st in split_toks:
                if not st:  # st may be '', e.g. tok = '*' then split_toks = ['', '']
                    continue
                return_tokens.append(st)
                new_tokens.append(st)
        return new_tokens

    def split_by_camel(token):
        camel_split = inflection.underscore(token).split('_')
        if len(camel_split) > 1:
            if any([len(cs) > 2 for cs in camel_split]):
                return_tokens.extend(camel_split)
                camel_word_split_record[token] = camel_split
                if not retain_camelcase:
                    return_tokens.remove(token)

    camel_word_split_record = {}  # record camel words and their generation e.g. CheckBuff: [check, buff]
    # return_tokens = tokens[:]
    return_tokens = []
    for tok in tokens:
        return_tokens.append(tok)
        if not bool(re.search(r'[a-zA-Z]', tok)):
            continue
        new_tokens = split_by_punc(tok)
        new_tokens = new_tokens if new_tokens else [tok]
        for nt in new_tokens:
            split_by_camel(nt)
    return return_tokens, camel_word_split_record


def normalize(tokens):
    normalized_tokens = [tok.lower() for tok in tokens]
    return normalized_tokens


def filter_words(tokens):
    tokens = [tok for tok in tokens if tok not in filter_words_set and len(tok) > 1]
    return tokens


def sentence_tokenize(document):
    sentences = sent_tokenize(document)
    return sentences


def cal_top(results):
    top1, top5, top10 = 0, 0, 0
    for r in results:
        first_buggy_idx = r.index('1') + 1
        if first_buggy_idx <= 10:
            top10 += 1
        if first_buggy_idx <= 5:
            top5 += 1
        if first_buggy_idx == 1:
            top1 += 1
    top1 = top1 / len(results)
    top5 = top5 / len(results)
    top10 = top10 / len(results)
    return top1, top5, top10


def cal_map(results):
    avg_p_list = []
    for r in results:
        buggy_indices = [idx + 1 for idx in range(len(r)) if r[idx] == '1']
        prec_k = []
        for ith_buggy, rank in enumerate(buggy_indices):
            prec_k.append((ith_buggy + 1) / rank)
        avg_p = sum(prec_k) / len(prec_k)
        avg_p_list.append(avg_p)
    MAP = sum(avg_p_list) / len(avg_p_list)
    return MAP


def cal_mrr(results):
    rr_list = []
    for r in results:
        first_buggy_idx = r.index('1') + 1
        rr_list.append(1 / first_buggy_idx)
    MRR = sum(rr_list) / len(rr_list)
    return MRR
