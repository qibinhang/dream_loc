import logging
import os
import sys
sys.path.append('..')
from configures import Configs
from Corpus.corpus import Corpus
from Features.collaborative_filtering import CollaborativeFiltering
from Features.cyclomatic_complexity import CyclomaticComplexity
from Features.fixing_history import FixingHistory
from Features.tfidf_plus_sim import *
from Features.trace import Trace
from utils import save_file
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
NUM_CPU = None


def generate_feature_collaborative_filtering(report_path, report_corpus,
                                             commit2code_paths, feature_dir):
    collaborative_filtering = CollaborativeFiltering(feature_dir)
    collaborative_filtering.collect(report_path, report_corpus, commit2code_paths)


def generate_feature_cyclomatic_complexity(code_collection, feature_dir):
    commit_path = code_collection['commit'] + '/' + code_collection['path']
    codes = list(zip(commit_path.tolist(), code_collection['content'].tolist()))
    cc = CyclomaticComplexity(feature_dir, num_cpu=NUM_CPU)
    cc.calculate_cyclomatic_complexity(codes)


def generate_feature_fixing_history(report_corpus, feature_dir):
    fixing_history = FixingHistory(feature_dir)
    fixing_history.collect(report_corpus)


def generate_feature_tfidf_plus_sim(report_corpus, code_corpus, feature_dir):
    tfidf_plus_sim = calculate_tfidf_plus_sim(report_corpus, code_corpus)
    logging.info(f'saving tfidf_plus_sim.pkl...')
    save_file(f'{feature_dir}/tfidf_plus_sim.pkl', tfidf_plus_sim)


def generate_feature_trace(commit2code_paths, report_corpus, feature_dir):
    trace = Trace(feature_dir)
    trace.collect(report_corpus, commit2code_paths)


def main(project_name):
    configs = Configs(project_name)
    logging.info(f"feature_dir: {configs.feature_dir}\n")
    global NUM_CPU
    NUM_CPU = configs.num_cpu

    if not os.path.exists(configs.feature_dir):
        os.mkdir(configs.feature_dir)

    corpus = Corpus(configs.corpus_dir)
    code_collection = corpus.load_collected_codes()
    code_corpus = corpus.load_code_corpus()
    report_corpus = corpus.load_report_corpus('total')
    commit2code_paths = corpus.load_commit2code_paths()
    logging.info('generating feature: collaborative_filtering...')
    generate_feature_collaborative_filtering(configs.report_path, report_corpus,
                                             commit2code_paths, configs.feature_dir)

    logging.info('generating feature: cyclomatic_complexity...')
    generate_feature_cyclomatic_complexity(code_collection, configs.feature_dir)

    logging.info('generating feature: fixing_history...')
    generate_feature_fixing_history(report_corpus, configs.feature_dir)

    logging.info('generating feature: tfidf_plus_sim...')
    generate_feature_tfidf_plus_sim(report_corpus, code_corpus, configs.feature_dir)

    # logging.info('generating feature: trace...')
    # generate_feature_trace(commit2code_paths, report_corpus, configs.feature_dir)


if __name__ == '__main__':
    project_name = sys.argv[1]
    main(project_name)
