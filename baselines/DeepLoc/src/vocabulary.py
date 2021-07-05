import pandas as pd
import sent2vec
import sys
import os
import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from configs import Configs
from utils import load_report_corpus, save_file, load_file, check_dir
from tqdm import tqdm


def generate_report_idx2desc_vec(max_num_sentence=50):
    bugid2report_idx = {}
    report_idx2desc_vec = []
    for idx, (_, row) in enumerate(report_corpus.iterrows()):
        bugid2report_idx[row['bug_id']] = idx
        sents = row['description'] if row['description'] else ' '
        sents = sents.split('\n')
        sents_vec = sent2vec_model.embed_sentences(sents)
        if len(sents) < max_num_sentence:
            sents_vec = np.concatenate([sents_vec, np.zeros((max_num_sentence-len(sents), 600))], axis=0)
        sents_vec = sents_vec[:max_num_sentence, :].reshape(-1)
        report_idx2desc_vec.append(sents_vec)

    assert len(report_idx2desc_vec) == report_corpus.shape[0]
    save_file(configs.bugid2idx_path, bugid2report_idx)
    save_file(configs.bugidx2desc_vec_path, np.array(report_idx2desc_vec))


def generate_report_idx2summary_vec(max_len=50):
    bugidx2summary_vec = []
    for idx, (_, row) in enumerate(report_corpus.iterrows()):
        summary = row['summary']
        vec = [word2idx[tok] for tok in summary.split()]
        vec = vec[:max_len]
        bugidx2summary_vec.append(vec)
    assert len(bugidx2summary_vec) == report_corpus.shape[0]
    pad_bugidx2summary_vec = list(itertools.zip_longest(*bugidx2summary_vec, fillvalue=0))
    pad_bugidx2summary_vec = np.array(pad_bugidx2summary_vec).transpose((1, 0))
    save_file(configs.bugidx2summary_vec_path, pad_bugidx2summary_vec)


def vectorize_code_corpus(max_num_line=300):
    sentences = code_corpus['content'].tolist()
    tfidf = TfidfVectorizer().fit(sentences)
    tfidf_word2idx = tfidf.vocabulary_
    tfidf_values = tfidf.transform(sentences)

    pad_line_vec = np.zeros(600)
    commit_code_path2code_idx = {}
    code_idx2line_idx = []
    line_idx2vec = [pad_line_vec]
    line_idx = 1
    for code_idx, (_, row) in enumerate(tqdm(code_corpus.iterrows(), total=code_corpus.shape[0],
                                             desc='vectorize code corpus', ncols=100)):
        commit, path, content = row['commit'], row['path'], row['content'].strip()
        commit_code_path2code_idx[f'{commit}/{path}'] = code_idx
        relative_line_idx = []
        if content:
            src_lines = content.split('\n')
            assert src_lines
            for line in src_lines:
                line_emb = []
                words = line.split()
                assert words
                for w in words:
                    tfidf_with_word_emb = tfidf_values[code_idx, tfidf_word2idx[w]] * word_idx2vec[word2idx[w]]
                    line_emb.append(tfidf_with_word_emb)
                line_emb = np.mean(np.array(line_emb), axis=0)

                relative_line_idx.append(line_idx)
                line_idx2vec.append(line_emb)
                line_idx += 1

        if len(relative_line_idx) < max_num_line:
            relative_line_idx += [0] * (max_num_line - len(relative_line_idx))
        code_idx2line_idx.append(relative_line_idx[:max_num_line])

    save_file(configs.code_commit_path2code_idx_path, commit_code_path2code_idx)
    save_file(configs.code_idx2line_idx_path, np.array(code_idx2line_idx))
    save_file(configs.line_idx2vec_path, np.array(line_idx2vec))


def generate_vocabulary():
    word2idx = {}
    idx2vec = []
    words = set()
    for _, item in code_corpus.iterrows():
        content = item['content']
        part_words = set(content.split())
        words = words.union(part_words)

    for _, item in report_corpus.iterrows():
        content = item['summary']
        part_words = set(content.split())
        words = words.union(part_words)

    word2idx['<pad>'] = 0
    idx2vec.append(np.zeros(600))
    for i, w in enumerate(words):
        word2idx[w] = i + 1
        if w in word2vec:
            idx2vec.append(word2vec[w])
        else:
            idx2vec.append(np.random.uniform(-0.25, 0.25, 600))
    idx2vec = np.array(idx2vec)
    save_file(configs.word2idx_path, word2idx)
    save_file(configs.word_idx2vec_path, idx2vec)


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    code_corpus = pd.read_csv(configs.code_corpus_path)
    code_corpus.fillna('', inplace=True)
    report_corpus = load_report_corpus(tag='total', report_corpus_dir=configs.report_corpus_dir)
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(f'{configs.data_dir}/wiki_unigrams.bin')
    uni_embs, vocab = sent2vec_model.get_unigram_embeddings()
    word2vec = {}
    for idx, w in enumerate(vocab):
        word2vec[w] = uni_embs[idx]

    check_dir(configs.vocabulary_dir)
    generate_report_idx2desc_vec()

    generate_vocabulary()
    word2idx = load_file(configs.word2idx_path)
    word_idx2vec = load_file(configs.word_idx2vec_path)
    generate_report_idx2summary_vec()
    vectorize_code_corpus()
