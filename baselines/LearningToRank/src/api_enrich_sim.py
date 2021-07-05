import sys
import pandas as pd
import numpy as np
from configs import Configs
from utils import load_file, load_report_corpus, save_file
from preprocess import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def generate_api_enrich_sim():
    src_api = load_file(configs.src_api_path)
    api_desc = pd.read_csv(configs.api_desc_path)
    api2desc = preprocess_api_desc(api_desc)
    report_corpus = load_report_corpus('total', configs.report_corpus_dir)
    commit2code_commit_paths = load_file(configs.commit2code_commit_paths_path)
    sim = _generate_api_enrich_sim(report_corpus, src_api, api2desc, commit2code_commit_paths)
    save_file(configs.api_enrich_sim_path, sim)


def preprocess_api_desc(api_desc):
    api2desc = {}
    for idx, row in api_desc.iterrows():
        tokens = preprocess(row['description'])
        api2desc[row['class_name']] = ' '.join(tokens)
    return api2desc


def _generate_api_enrich_sim(report_corpus, src_api, api2desc, commit2code_commit_paths):
    api_enrich_sim = {}
    for _, row in tqdm(report_corpus.iterrows(), desc='api_enrich_sim', total=report_corpus.shape[0], ncols=100):
        bug_id = row['bug_id']
        commit = row['commit']
        content = f"{row['summary']} {row['description']}"
        relative_src_commit_path = list(commit2code_commit_paths[commit].values())
        relative_src_api = dict([(commit_path, src_api[commit_path]) for commit_path in relative_src_commit_path])
        relative_src_api_desc = {}
        for commit_path in relative_src_api:
            apis = relative_src_api[commit_path]
            api_desc_list = []
            for api in apis:
                if api in api2desc:
                    api_desc_list.append(api2desc[api])
            relative_src_api_desc[commit_path] = api_desc_list
        similarity = cal_sim(content, relative_src_api_desc)
        api_enrich_sim[bug_id] = similarity
    return api_enrich_sim


def cal_sim(report_content, relative_src_api_desc):
    similarity = {}
    api_desc_sentences = []
    for each_src in relative_src_api_desc.values():
        api_desc_sentences += each_src
    report_sentence = [report_content]
    sentences = api_desc_sentences + report_sentence
    tfidf = TfidfVectorizer().fit(sentences)
    for commit_path in relative_src_api_desc:
        src_api_desc = relative_src_api_desc[commit_path]
        if src_api_desc:
            cat_api = src_api_desc + [' '.join(src_api_desc)]
            report_tfidf = tfidf.transform(report_sentence)
            api_tfidf = tfidf.transform(cat_api)
            sim = cosine_similarity(report_tfidf, api_tfidf)
            sim = np.max(sim)
        else:
            sim = 0.0
        similarity[commit_path] = sim
    return similarity


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    generate_api_enrich_sim()
