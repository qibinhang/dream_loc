import sys
import numpy as np
from configs import Configs
from utils import load_code_corpus, load_report_corpus, load_file, save_file
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def generate_surface_sim():
    sur_sim = {}
    for _, item in tqdm(report_corpus.iterrows(), desc='cal surface_sim', total=report_corpus.shape[0], ncols=100):
        bugid, commit, content = item['bug_id'], item['commit'], f"{item['summary']} {item['description']}"
        relative_code_commit_path = list(commit2commit_path[commit].values())
        relative_code_idx = [commit_path2idx[commit_path] for commit_path in relative_code_commit_path]
        relative_code_content = [' '.join(code_corpus[idx]) for idx in relative_code_idx]
        relative_tfidf = cal_tfidf([content], relative_code_content)
        relative_tfidf = np.squeeze(relative_tfidf)
        relative_tfidf = relative_tfidf.tolist()
        relative_tfidf = list(zip(relative_code_commit_path, relative_tfidf))
        sur_sim[bugid] = relative_tfidf
    save_file(configs.surface_sim_path, sur_sim)


def cal_tfidf(report, codes):
    r_tfidf = tfidf.transform(report)
    c_tfidf = tfidf.transform(codes)
    sim = cosine_similarity(r_tfidf, c_tfidf)
    return sim


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    code_corpus = load_code_corpus(configs.code_corpus_dir)
    report_corpus = load_report_corpus(tag='train', report_corpus_dir=configs.report_corpus_dir)
    report_sentences = [f"{row['summary']} {row['description']}" for _, row in report_corpus.iterrows()]
    tfidf = TfidfVectorizer().fit(report_sentences)
    vocab = tfidf.vocabulary_
    save_file(configs.sur_sim_vocab_path, vocab)

    code_sentences = [' '.join(each_code) for each_code in code_corpus]

    tfidf = TfidfVectorizer(vocabulary=vocab).fit(report_sentences + code_sentences)

    report_corpus = load_report_corpus(tag='total', report_corpus_dir=configs.report_corpus_dir)
    commit_path2idx = load_file(configs.commit_path2idx_path)
    commit2commit_path = load_file(configs.commit2code_commit_paths_path)

    generate_surface_sim()
