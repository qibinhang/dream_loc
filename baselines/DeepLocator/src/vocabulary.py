import itertools
import sys
import pandas as pd
import numpy as np
from configs import Configs
from utils import load_file, load_report_corpus, save_file
from gensim.models import KeyedVectors
from tqdm import tqdm


def count_oov():
    report_oov = set()
    code_oov = set()
    for _, item in code_corpus.iterrows():
        content = item['content']
        words = content.split()
        for each_w in words:
            if each_w not in word2vec:
                code_oov.add(each_w)

    for _, item in report_corpus.iterrows():
        content = f"{report_corpus['summary']} {report_corpus['description']}"
        words = content.split()
        for each_w in words:
            if each_w not in word2vec:
                report_oov.add(each_w)

    print(f'report_oov: {len(report_oov)}')
    print(f'code_oov: {len(code_oov)}')


def generate_vocabulary():
    word2idx = {}
    idx2vec = []
    words = set()
    for _, item in code_corpus.iterrows():
        content = item['content']
        part_words = set(content.split())
        words = words.union(part_words)

    for _, item in report_corpus.iterrows():
        content = f"{item['summary']} {item['description']}"
        part_words = set(content.split())
        words = words.union(part_words)

    word2idx['<pad>'] = 0
    idx2vec.append(np.zeros(100))
    for i, w in enumerate(words):
        word2idx[w] = i + 1
        if w in word2vec:
            idx2vec.append(word2vec[w])
        else:
            idx2vec.append(np.random.uniform(-0.25, 0.25, 100))
    idx2vec = np.array(idx2vec)
    save_file(configs.word2idx_path, word2idx)
    save_file(configs.word_idx2vec_path, idx2vec)


def vectorize_report_corpus(max_len=300):
    bugid2idx = {}
    bugidx2vec = []
    for idx, (_, item) in enumerate(tqdm(report_corpus.iterrows(), desc='report vec', ncols=100, total=report_corpus.shape[0])):
        bugid = item['bug_id']
        content = f"{item['summary']} {item['description']}"
        vec = sentence2vec(content)
        vec = vec[:max_len]
        bugid2idx[bugid] = idx
        bugidx2vec.append(vec)

    pad_bugidx2vec = list(itertools.zip_longest(*bugidx2vec, fillvalue=0))
    pad_bugidx2vec = np.array(pad_bugidx2vec).transpose((1, 0))

    save_file(configs.bugid2idx_path, bugid2idx)
    save_file(configs.bugidx2vec_path, pad_bugidx2vec)


def vectorize_code_corpus(max_len=500):
    commit_path2idx = {}
    code_idx2vec = []
    for idx, (_, item) in enumerate(tqdm(code_corpus.iterrows(), desc='code vec', ncols=100, total=code_corpus.shape[0])):
        commit, path, content = item['commit'], item['path'], item['content']
        vec = sentence2vec(content)
        vec = vec[:max_len]
        code_idx2vec.append(vec)
        commit_path2idx[f'{commit}/{path}'] = idx

    pad_code_idx2vec = list(itertools.zip_longest(*code_idx2vec, fillvalue=0))
    pad_code_idx2vec = np.array(pad_code_idx2vec).transpose((1, 0))

    save_file(configs.commit_path2code_idx_path, commit_path2idx)
    save_file(configs.code_idx2vec_path, pad_code_idx2vec)


def sentence2vec(sentence):
    tokens = sentence.split()
    vec = [word2idx[tok] for tok in tokens]
    return vec


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    code_corpus = pd.read_csv(configs.code_corpus_path)
    code_corpus.fillna('', inplace=True)

    report_corpus = load_report_corpus('total', configs.report_corpus_dir)
    word2vec = KeyedVectors.load_word2vec_format(configs.word2vec_path, binary=False)
    generate_vocabulary()
    word2idx = load_file(configs.word2idx_path)
    vectorize_report_corpus()
    vectorize_code_corpus()
