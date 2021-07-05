import pandas as pd
import sys
import os
from configs import Configs
from utils import load_report, load_file, tokenize, split_camelcase, normalize, filter_words, sentence_tokenize
from tqdm import tqdm


def generate_report_corpus(reports, tag):
    assert tag in ('train', 'val', 'test')
    corpus = []
    for r in tqdm(reports, ncols=80):
        commit = r[7].text
        buggy_file_paths = r[9].text.split('\n')
        buggy_file_paths = filter_buggy_path(buggy_file_paths, commit)
        if not buggy_file_paths:
            continue

        bug_id = r[1].text
        summary = r[2].text
        description = r[3].text if r[3].text else ''
        report_timestamp = int(r[5].text)
        commit_timestamp = int(r[8].text)

        summary = preprocess_summary(summary)
        description = preprocess_description(description)

        each_row = [
            bug_id, commit, summary, description, '\n'.join(buggy_file_paths), report_timestamp, commit_timestamp
        ]
        corpus.append(each_row)
    columns = ['bug_id', 'commit', 'summary', 'description', 'buggy_paths', 'report_timestamp', 'commit_timestamp']
    df = pd.DataFrame(data=corpus, columns=columns)
    df.to_csv(f'{configs.report_corpus_dir}/{tag}_report_corpus.csv')
    return corpus


def preprocess_summary(summary):
    tokens = _preprocess(summary)
    tokens = tokens[1:]
    return ' '.join(tokens)


def preprocess_description(description):
    sentences = []
    sents = sentence_tokenize(description)
    for s in sents:
        tokens = _preprocess(s)
        sentences.append(' '.join(tokens))
    return '\n'.join(sentences)


def _preprocess(sentence):
    tokens = tokenize(sentence)
    split_tokens, _ = split_camelcase(tokens)
    normalized_tokens = normalize(split_tokens)
    filter_tokens = filter_words(normalized_tokens)
    return filter_tokens


def filter_buggy_path(buggy_file_paths, commit):
    """filter out buggy files that are 'ADD'"""
    valid_paths = [path for path in buggy_file_paths if path in commit2path2commit_path[commit]]
    return valid_paths


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)

    commit2path2commit_path = load_file(configs.commit2code_commit_paths_path)
    train_reports, val_reports, test_reports = load_report(configs.report_path, tag='split')
    if not os.path.exists(configs.report_corpus_dir):
        os.makedirs(configs.report_corpus_dir)

    generate_report_corpus(train_reports, 'train')
    generate_report_corpus(val_reports, 'val')
    generate_report_corpus(test_reports, 'test')
