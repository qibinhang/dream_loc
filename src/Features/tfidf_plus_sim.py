import numpy as np
import sys
sys.path.append('..')
from collections import namedtuple
from evaluator import Evaluator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_file
Formatted_pred = namedtuple('Formatted_pred', 'pred buggy_code_paths')


def calculate_tfidf_plus_sim(report_corpus, code_corpus):
    """
    :return np.array, shape: (num_report, num_code), order: row=report_corpus column=code_corpus
    """
    report_sentences = [f'{r.description} {r.summary}' for r in report_corpus.itertuples()]
    code_sentences = code_corpus['snippets'].tolist()
    tfidf = TfidfVectorizer().fit(report_sentences + code_sentences)
    vocab = tfidf.vocabulary_
    idf = tfidf.idf_
    r_tfidf = tfidf.transform(report_sentences)
    c_tfidf = tfidf.transform(code_sentences)

    report_keywords = [r.keywords_summary.split() + r.keywords_description.split() for r in report_corpus.itertuples()]
    report_keywords_indices = [[vocab[w] for w in keywords] for keywords in report_keywords]
    report_keywords_indices_set = []
    for indices in report_keywords_indices:
        report_keywords_indices_set += indices
    report_keywords_indices_set = set(report_keywords_indices_set)
    keywords_plus_ratio = cal_keyword_plus_ratio(report_keywords_indices_set, idf)
    report_plus_tfidf = cal_tfidf_plus(r_tfidf, report_keywords_indices, keywords_plus_ratio)

    tfidf_plus_sim = cosine_similarity(report_plus_tfidf, c_tfidf)
    return tfidf_plus_sim


def cal_keyword_plus_ratio(keywords_indices, idx2idf):
    keywords_idf = np.array([idx2idf[idx] for idx in keywords_indices])
    keywords_idf = (keywords_idf - keywords_idf.min()) / (keywords_idf.max() - keywords_idf.min())
    keywords_ratio = keywords_idf + 1
    keywords_ratio = dict(list(zip(keywords_indices, keywords_ratio)))
    return keywords_ratio


def cal_tfidf_plus(tfidf, keywords_indices, keyword_plus_ratio):
    for i, indices in enumerate(keywords_indices):
        for idx in indices:
            tfidf[i, idx] = tfidf[i, idx] * keyword_plus_ratio[idx]
    return tfidf


def load_tfidf_plus_sim(feature_dir):
    tfidf_plus_sim = load_file(f'{feature_dir}/tfidf_plus_sim.pkl')
    return tfidf_plus_sim


def evaluate(report_corpus, tfidf_plus_sim):
    print()
    all_format_pred = []
    for report in report_corpus:
        bug_id = report.bug_id
        buggy_paths = report.buggy_path.split('\n')
        all_format_pred.append(
            Formatted_pred(pred=list(tfidf_plus_sim[bug_id].items()), buggy_code_paths=buggy_paths)
        )
    evaluator = Evaluator()
    ranked_predict = evaluator.rank(all_format_pred)
    hit_k, mean_ap, mean_rr = evaluator.evaluate(ranked_predict)

    print(f'MAP:   {mean_ap:.4f}')
    print(f'MRR:   {mean_rr:.4f}')
    for n, hit in enumerate(hit_k):
        print(f'hit_{n + 1}: {hit:.4f}')
