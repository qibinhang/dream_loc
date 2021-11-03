import logging
import os
import random
import sys
import _pickle as pickle
from configures import Configs
from Corpus.corpus import Corpus
from itertools import cycle
from vocabulary import load_report_corpus_vectors, load_bugidx2path2idx
from tqdm import tqdm
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def generate_train_dataset(report_corpus, bugid2idx, bugidx2path2idx, dataset_dir, num_neg, tag):
    random.seed(1)
    dataset = []
    for report in tqdm(report_corpus.itertuples(), total=report_corpus.shape[0], ncols=80):
        report_idx = bugid2idx[report.bug_id]
        path2idx = bugidx2path2idx[report_idx]
        buggy_path = report.buggy_paths.split('\n')
        all_path = set(path2idx.keys())
        assert set(buggy_path).issubset(all_path)
        normal_path = list(all_path - set(buggy_path))
        random.shuffle(normal_path)
        if num_neg < len(buggy_path):
            neg_path = normal_path[:len(buggy_path)]
        else:
            neg_path = normal_path[:num_neg]

        buggy_path_idx = [path2idx[path] for path in buggy_path]
        neg_path_idx = [path2idx[path] for path in neg_path]
        each_data = list(zip(cycle([report_idx]), cycle(buggy_path_idx), neg_path_idx))
        dataset += each_data
    with open(f'{dataset_dir}/dataset_for_{tag}.pkl', 'wb') as f:
        pickle.dump(dataset, f)


def generate_test_dataset(report_corpus, bugid2idx, bugidx2path2idx, dataset_dir, tag):
    dataset = []
    for report in tqdm(report_corpus.itertuples(), total=report_corpus.shape[0], ncols=80):
        report_idx = bugid2idx[report.bug_id]
        path2idx = bugidx2path2idx[report_idx]
        buggy_path = report.buggy_paths.split('\n')
        all_path = list(path2idx.keys())
        all_path_idx = list(path2idx.values())

        data = list(zip(cycle([report_idx]), all_path_idx))
        dataset += [(data, all_path, buggy_path)]
    with open(f'{dataset_dir}/dataset_for_{tag}.pkl', 'wb') as f:
        pickle.dump(dataset, f)


def load_dataset(dataset_dir, tag):
    assert tag in ('train', 'val_metric', 'test', 'val_loss')
    with open(f'{dataset_dir}/dataset_for_{tag}.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def main(project_name):
    configs = Configs(project_name)
    num_neg = configs.num_neg_sample
    logging.info(f"feature_dir: {configs.feature_dir}\n")

    corpus_dir = configs.corpus_dir
    vocab_dir = configs.vocab_dir
    dataset_dir = configs.dataset_dir
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    logging.info(f'dataset dir: {dataset_dir}')
    print(f'num_neg = {configs.num_neg_sample}')

    corpus = Corpus(corpus_dir)
    _, bugid2idx = load_report_corpus_vectors(save_dir=vocab_dir)
    bugidx2path2idx = load_bugidx2path2idx(vocab_dir)

    tag = 'train'
    logging.info(f'generating {tag} dataset...')
    report_corpus = corpus.load_report_corpus(tag)
    generate_train_dataset(report_corpus, bugid2idx, bugidx2path2idx, dataset_dir, num_neg=num_neg, tag='train')

    tag = 'val'
    logging.info(f'generating {tag} dataset for MAP/MRR/TOP metric...')
    report_corpus = corpus.load_report_corpus(tag)
    generate_test_dataset(report_corpus, bugid2idx, bugidx2path2idx, dataset_dir, tag='val_metric')

    logging.info(f'generation {tag} dataset for val_loss')
    generate_train_dataset(report_corpus, bugid2idx, bugidx2path2idx, dataset_dir, num_neg=num_neg, tag='val_loss')

    tag = 'test'
    logging.info(f'generating {tag} dataset...')
    report_corpus = corpus.load_report_corpus(tag)
    generate_test_dataset(report_corpus, bugid2idx, bugidx2path2idx, dataset_dir, tag)


if __name__ == '__main__':
    project_name = sys.argv[1]
    main(project_name)
