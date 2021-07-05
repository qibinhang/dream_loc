import sys
import pandas as pd
from configs import Configs
from utils import load_file, load_report_corpus, save_file, check_dir
from tqdm import tqdm


def trans_format_for_svm(tag, top_n=-1):
    """
    1 qid:1 1:1 2:1 3:0 4:0.2 5:0 6:0 # 10526
    0 qid:1 1:0 2:0 3:1 4:0.1 5:1 6:0 # 10526
    0 qid:1 1:0 2:1 3:0 4:0.4 5:0 6:0 # 10526
    0 qid:1 1:0 2:0 3:1 4:0.3 5:0 6:0 # 10526
    0 qid:2 1:0 2:0 3:1 4:0.2 5:0 6:0 # 10972
    1 qid:2 1:1 2:0 3:1 4:0.4 5:0 6:0 # 10972
    .....
    """
    dataset = load_file(f'{configs.dataset_dir}/{tag}_dataset.pkl')
    trans_dataset = []
    for qid, items in enumerate(tqdm(dataset, desc=f'trans {tag}', ncols=100)):
        bugid = items[0]
        all_src_info = [dict(each_src_info) for each_src_info in items[1]]
        data = _trans_data_for_one_report(qid, bugid, all_src_info, top_n)
        trans_dataset += data

    with open(f'{configs.dataset_dir}/{tag}.dat', 'w') as f:
        for data in trans_dataset:
            f.write(f'{data}\n')


def _trans_data_for_one_report(qid, bugid, all_src_info, top_n):
    sorted_label_src = list(sorted(all_src_info, key=lambda x: x['label'], reverse=True))
    n_buggy = sum([1 for src_info in sorted_label_src if src_info['label'] == 1])
    assert n_buggy > 0

    normal_src = sorted_label_src[n_buggy:]
    sorted_normal_src = list(sorted(normal_src, key=lambda x: x['sur_sim'], reverse=True))
    if top_n <= 0:
        top_normal_src = sorted_normal_src
    else:
        top_normal_src = sorted_normal_src[:top_n]

    final_src = sorted_label_src[:n_buggy] + top_normal_src

    # scaling
    sur_sim, api_sim, name_sim, cf_value, recency_value, frequency_value = [], [], [], [], [], []
    for src_info in all_src_info:
        sur_sim.append(src_info['sur_sim'])
        api_sim.append(src_info['api_sim'])
        name_sim.append(src_info['name_sim'])
        cf_value.append(src_info['cf_value'])
        recency_value.append(src_info['recency_value'])
        frequency_value.append(src_info['frequency_value'])

    max_sur, min_sur = max(sur_sim), min(sur_sim)
    max_api, min_api = max(api_sim), min(api_sim)
    max_name, min_name = max(name_sim), min(name_sim)
    max_cf, min_cf = max(cf_value), min(cf_value)
    max_rv, min_rv = max(recency_value), min(recency_value)
    max_fv, min_fv = max(frequency_value), min(frequency_value)

    data = []
    for src_info in final_src:
        one_data = f"{src_info['label']} qid:{qid} " \
                   f"1:{cal_scaling(src_info['sur_sim'], max_sur, min_sur):.4f} " \
                   f"2:{cal_scaling(src_info['api_sim'], max_api, min_api):.4f} " \
                   f"3:{cal_scaling(src_info['name_sim'], max_name, min_name):.4f} " \
                   f"4:{cal_scaling(src_info['cf_value'], max_cf, min_cf):.4f} " \
                   f"5:{cal_scaling(src_info['recency_value'], max_rv, min_rv):.4f} " \
                   f"6:{cal_scaling(src_info['frequency_value'], max_fv, min_fv):.4f} " \
                   f"# {bugid}"
        data.append(one_data)
    return data


def cal_scaling(value, max_v, min_v):
    if value - min_v == 0:
        return 0
    else:
        return (value - min_v) / (max_v - min_v)


def generate_dataset(tag):
    assert tag in ('train', 'test')
    if tag == 'train':
        report_corpus = load_report_corpus(tag, configs.report_corpus_dir)
    else:
        report_corpus_1 = load_report_corpus('val', configs.report_corpus_dir)
        report_corpus_2 = load_report_corpus('test', configs.report_corpus_dir)
        report_corpus = pd.concat([report_corpus_1, report_corpus_2])
    dataset = []

    for _, row in tqdm(report_corpus.iterrows(), desc=f'{tag} dataset', total=report_corpus.shape[0], ncols=100):
        bugid = row['bug_id']
        commit = row['commit']
        buggy_paths = row['buggy_paths'].split('\n')
        relative_src_path2commit_path = commit2code_commit_paths[commit]
        data_for_each_report = []
        check_bug_exist_flag = 0
        for src_path in relative_src_path2commit_path:
            commit_path = relative_src_path2commit_path[src_path]
            sur_sim = get_sur_sim(bugid, commit_path)
            api_sim = get_api_sim(bugid, commit_path)
            name_sim = get_class_name_sim(bugid, commit_path)
            cf_value = get_collaborative_filtering(str(bugid), src_path)
            recency_value = get_fixing_recency(str(bugid), src_path)
            frequency_value = get_fixing_frequency(str(bugid), src_path)
            if src_path in buggy_paths:
                label = 1
                check_bug_exist_flag = 1
            else:
                label = 0
            data_for_each_report.append(
                [('src_path', src_path), ('label', label), ('sur_sim', sur_sim), ('api_sim', api_sim),
                 ('name_sim', name_sim), ('cf_value', cf_value),
                 ('recency_value', recency_value), ('frequency_value', frequency_value)]
            )
        if check_bug_exist_flag == 0:
            raise ValueError(f'{bugid} has no buggy files?')
        dataset.append((bugid, data_for_each_report))
    check_dir(configs.dataset_dir)
    save_file(f'{configs.dataset_dir}/{tag}_dataset.pkl', dataset)


def get_sur_sim(bugid, src_commit_path):
    bug_idx = bugid2idx[bugid]
    code_idx = commit_path2idx[src_commit_path]
    sim = surface_sim[bug_idx][code_idx]
    return sim


def get_api_sim(bugid, src_commit_path):
    sim = api_enrich_sim[bugid][src_commit_path]
    return sim


def get_class_name_sim(bugid, src_commit_path):
    sim = class_name_sim[bugid][src_commit_path]
    return sim


def get_collaborative_filtering(bugid, src_path):
    value = collaborative_filtering[bugid].get(src_path, 0)
    return value


def get_fixing_recency(bugid, src_path):
    value = fixing_recency[bugid].get(src_path, 0)
    return value


def get_fixing_frequency(bugid, src_path):
    value = fixing_frequency[bugid].get(src_path, 0)
    return value


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    surface_sim = load_file(configs.surface_sim_path)
    api_enrich_sim = load_file(configs.api_enrich_sim_path)
    collaborative_filtering = load_file(configs.collaborative_filtering_path)
    class_name_sim = load_file(configs.class_name_sim_path)
    fixing_frequency = load_file(configs.fixing_frequency_path)
    fixing_recency = load_file(configs.fixing_recency_path)

    commit2code_commit_paths = load_file(configs.commit2code_commit_paths_path)
    bugid2idx = load_file(configs.bugid2idx_path)
    bugidx2path_idx2valid_path_idx = load_file(configs.bugidx2path_idx2valid_path_idx_path)
    commit_path2idx = load_file(configs.commit_path2idx_path)
    generate_dataset('train')
    generate_dataset('test')

    trans_format_for_svm('train', top_n=300)
    trans_format_for_svm('test')

