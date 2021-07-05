import sys
import random
from configs import Configs
from utils import load_file, load_report_corpus, check_dir, save_file
from tqdm import tqdm
random.seed(63)


def generate_dataset(tag, num_neg=300):
    assert tag in ('train', 'val', 'test')
    report_corpus = load_report_corpus(tag, configs.report_corpus_dir)

    dataset = []
    for _, row in tqdm(report_corpus.iterrows(), desc=f'{tag} dataset', total=report_corpus.shape[0], ncols=100):
        bugid = row['bug_id']
        bug_idx = bugid2idx[bugid]
        commit = row['commit']
        buggy_paths = row['buggy_paths'].split('\n')
        relative_src_path2commit_path = commit2code_commit_paths[commit]
        data_for_each_report = []
        check_bug_exist_flag = 0
        neg_src_path = get_neg_samples(list(total_neg_path_and_commit_path.keys()),
                                       buggy_paths,
                                       num_neg)

        samples_src_path = buggy_paths + neg_src_path
        samples_src_path_set = set(samples_src_path)
        assert len(samples_src_path_set) == len(samples_src_path)
        for src_path in samples_src_path:
            if src_path in buggy_paths:
                recency_value = get_fixing_recency(str(bugid), src_path)
                frequency_value = get_fixing_frequency(str(bugid), src_path)
                commit_path = relative_src_path2commit_path[src_path]
            else:  # for neg sample
                recency_value = get_fixing_recency(str(neg_from_bugid), src_path)
                frequency_value = get_fixing_frequency(str(neg_from_bugid), src_path)
                commit_path = total_neg_path_and_commit_path[src_path]
            code_idx = commit_path2idx[commit_path]
            if src_path in buggy_paths:
                label = 1
                check_bug_exist_flag = 1
            else:
                label = 0
            data_for_each_report.append((bug_idx, code_idx, recency_value, frequency_value, label))

        if check_bug_exist_flag == 0:
            raise ValueError(f'{bugid} has no buggy files?')

        dataset.append(data_for_each_report)

    check_dir(configs.dataset_dir)
    save_file(f'{configs.dataset_dir}/{tag}_dataset_neg_{configs.num_neg}.pkl', dataset)


def get_neg_samples(total_neg_path, pos_samples, num_neg):
    neg_samples = []
    random.shuffle(total_neg_path)
    for path in total_neg_path:
        if path not in pos_samples:
            neg_samples.append(path)
            if len(neg_samples) == num_neg:
                break
    return neg_samples


def get_fixing_recency(bugid, src_path):
    value = fixing_recency[bugid].get(src_path, 0)
    return value


def get_fixing_frequency(bugid, src_path):
    value = fixing_frequency[bugid].get(src_path, 0)
    return value


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    print(f'num_neg: {configs.num_neg}')

    fixing_frequency = load_file(configs.fixing_frequency_path)
    fixing_recency = load_file(configs.fixing_recency_path)

    commit2code_commit_paths = load_file(configs.commit2code_commit_paths_path)
    bugid2idx = load_file(configs.bugid2idx_path)
    commit_path2idx = load_file(configs.commit_path2code_idx_path)

    neg_from_commit = list(commit2code_commit_paths.keys())[-1]
    total_neg_path_and_commit_path = commit2code_commit_paths[neg_from_commit]
    neg_from_bugid = -1
    total_report_corpus = load_report_corpus('total', configs.report_corpus_dir)
    for _, row in total_report_corpus.iterrows():
        bugid = row['bug_id']
        commit = row['commit']
        if commit == neg_from_commit:
            neg_from_bugid = bugid
            break

    generate_dataset('train', num_neg=configs.num_neg)
    generate_dataset('val', num_neg=-1)
    generate_dataset('test', num_neg=-1)
