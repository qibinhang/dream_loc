import os
import pandas as pd
import _pickle as pickle
from sklearn.externals import joblib


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


def load_report_corpus(tag, report_corpus_dir):
    assert tag in ('train', 'val', 'test', 'total')
    if tag == 'total':
        report_corpus = _load_report_corpus('train', report_corpus_dir)
        report_corpus = pd.concat([report_corpus, _load_report_corpus('val', report_corpus_dir)])
        report_corpus = pd.concat([report_corpus, _load_report_corpus('test', report_corpus_dir)])
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

