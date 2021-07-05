import numpy as np
import sys
from configs import Configs
from gensim.models import KeyedVectors
from utils import load_file, load_report_corpus, load_code_corpus, save_file
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def generate_embedding_sim():
    embedding_similarity = {}
    for _, row in tqdm(report_corpus.iterrows(), desc=f'embedding_sim', total=report_corpus.shape[0], ncols=100):
        bugid = row['bug_id']
        content = f"{row['summary']} {row['description']}".split()
        commit = row['commit']
        relative_src_commit_path = list(commit2code_commit_paths[commit].values())
        similarity = _gen_embedding_sim_for_one_report(content, relative_src_commit_path)
        embedding_similarity[bugid] = similarity
    return embedding_similarity


def _gen_embedding_sim_for_one_report(report_content, relative_src_path2commit_path):
    similarity = {}
    for commit_path in relative_src_path2commit_path:
        src_idx = commit_path2idx[commit_path]
        # code_vectors = code2vectors(code_corpus[src_idx])
        code_vectors = code_corpus_vectors[src_idx]
        report_vectors = np.array(sentence2vector(report_content))
        asy_sim_1, asy_sim_2 = cal_asymmetric_sim(report_vectors, code_vectors)
        similarity[commit_path] = (asy_sim_1, asy_sim_2)
    return similarity


def cal_asymmetric_sim(report_vectors, code_vectors):
    sim = cosine_similarity(report_vectors, code_vectors)
    max_sim_1 = np.max(sim, axis=1)
    max_sim_1 = max_sim_1[max_sim_1 > 0]
    if max_sim_1.size > 0:
        asy_sim_1 = sum(max_sim_1) / len(max_sim_1)
    else:
        asy_sim_1 = 0

    max_sim_2 = np.max(sim, axis=0)
    max_sim_2 = max_sim_2[max_sim_2 > 0]
    if max_sim_2.size > 0:
        asy_sim_2 = sum(max_sim_2) / len(max_sim_2)
    else:
        asy_sim_2 = 0
    return asy_sim_1, asy_sim_2


# use sur_sim_vocab
def sentence2vector(sentence):
    sentence_vec = []
    for token in sentence:
        if token in word_vocab:
            if token in word2vec:
                sentence_vec.append(word2vec[token])
    if not sentence_vec:
        sentence_vec.append(np.zeros(100))
    return sentence_vec


if __name__ == '__main__':
    project_name = sys.argv[1]

    configs = Configs(project_name)

    report_corpus = load_report_corpus(tag='train', report_corpus_dir=configs.report_corpus_dir)
    report_sentences = [f"{row['summary']} {row['description']}" for _, row in report_corpus.iterrows()]
    tfidf = TfidfVectorizer().fit(report_sentences)
    word_vocab = tfidf.vocabulary_

    report_corpus = load_report_corpus('total', configs.report_corpus_dir)
    code_corpus = load_code_corpus(configs.code_corpus_dir)
    commit_path2idx = load_file(configs.commit_path2idx_path)
    commit2code_commit_paths = load_file(configs.commit2code_commit_paths_path)
    word2vec = KeyedVectors.load_word2vec_format(configs.word2vec_path, binary=False)

    code_corpus_vectors = []
    for code in tqdm(code_corpus, desc='trans code_corpus to vectors', ncols=100):
        each_code_vectors = sentence2vector(code)
        code_corpus_vectors.append(np.array(each_code_vectors))

    embedding_sim = generate_embedding_sim()
    save_file(configs.embedding_sim_path, embedding_sim)
