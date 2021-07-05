import sys
import os
from configs import Configs
from utils import load_file, load_report_corpus, save_file
from tqdm import tqdm


def generate_class_name_sim():
    report_corpus = load_report_corpus('total', configs.report_corpus_dir)
    commit2code_commit_paths = load_file(configs.commit2code_commit_paths_path)
    class_name_sim = {}
    for _, row in tqdm(report_corpus.iterrows(), desc='class_name_sim', total=report_corpus.shape[0], ncols=100):
        bug_id = row['bug_id']
        commit = row['commit']
        summary = row['summary']
        relative_src = commit2code_commit_paths[commit].values()
        sim = cal_sim(summary, relative_src)
        class_name_sim[bug_id] = sim
    save_file(configs.class_name_sim_path, class_name_sim)


def cal_sim(report_summary, relative_src):
    sim = {}
    for commit_path in relative_src:
        main_class_name = os.path.basename(commit_path)
        main_class_name = main_class_name[:-5]
        if main_class_name.lower() in report_summary.split():
            sim[commit_path] = len(main_class_name)
        else:
            sim[commit_path] = 0
    return sim


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    generate_class_name_sim()
