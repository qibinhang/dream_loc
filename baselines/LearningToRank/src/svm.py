import sys
import os
import random
from configs import Configs
from tqdm import tqdm
random.seed(0)


def train():
    cmd = f'./svm/svm_rank_learn -v 0 -c 0.01 {configs.dataset_dir}/train.dat {configs.svm_result_dir}/models/model'
    os.system(cmd)


def predict(tag):
    cmd = f'./svm/svm_rank_classify -v 0  {configs.dataset_dir}/{tag}.dat {configs.svm_result_dir}/models/model ' \
          f'{configs.svm_result_dir}/predicts/{tag}.txt'
    os.system(cmd)


def evaluation(tag):
    predict(tag)
    with open(f'{configs.svm_result_dir}/predicts/{tag}.txt', 'r') as f:
        pred = f.read().strip().split('\n')
    with open(f'{configs.dataset_dir}/{tag}.dat', 'r') as f:
        dataset = f.read().strip().split('\n')
    assert len(pred) == len(dataset)

    reformat_result = []
    each_pid_data = []
    last_pid = ' '
    for i, data in enumerate(dataset):
        data = data.split(' ')
        pid = data[1]
        if last_pid == ' ':
            last_pid = pid
        elif pid != last_pid:
            last_pid = pid
            random.shuffle(each_pid_data)
            sorted_labels = list(sorted(each_pid_data, key=lambda x: float(x[1]), reverse=True))
            sorted_labels = [item[0] for item in sorted_labels]
            reformat_result.append(sorted_labels)
            each_pid_data = []

        label = data[0]
        each_pid_data.append((label, pred[i]))
    top_1, top_5, top_10 = cal_top(reformat_result)
    MAP, MRR = cal_map(reformat_result), cal_mrr(reformat_result)
    return top_1, top_5, top_10, MAP, MRR


def cal_top(results):
    top1, top5, top10 = 0, 0, 0
    for r in results:
        first_buggy_idx = r.index('1') + 1
        if first_buggy_idx <= 10:
            top10 += 1
        if first_buggy_idx <= 5:
            top5 += 1
        if first_buggy_idx == 1:
            top1 += 1
    top1 = top1 / len(results)
    top5 = top5 / len(results)
    top10 = top10 / len(results)
    return top1, top5, top10


def cal_map(results):
    avg_p_list = []
    for r in results:
        buggy_indices = [idx + 1 for idx in range(len(r)) if r[idx] == '1']
        prec_k = []
        for ith_buggy, rank in enumerate(buggy_indices):
            prec_k.append((ith_buggy + 1) / rank)
        avg_p = sum(prec_k) / len(prec_k)
        avg_p_list.append(avg_p)
    MAP = sum(avg_p_list) / len(avg_p_list)
    return MAP


def cal_mrr(results):
    rr_list = []
    for r in results:
        first_buggy_idx = r.index('1') + 1
        rr_list.append(1 / first_buggy_idx)
    MRR = sum(rr_list) / len(rr_list)
    return MRR


if __name__ == '__main__':
    project_name = sys.argv[1]
    for_test = False
    if len(sys.argv[1]) == 3:
        for_test = True

    configs = Configs(project_name)
   
    if not os.path.exists(f'{configs.svm_result_dir}/models'):
        os.makedirs(f'{configs.svm_result_dir}/models')
        os.makedirs(f'{configs.svm_result_dir}/predicts')
    
    if for_test:
        top_1, top_5, top_10, MAP, MRR = evaluation(tag='test')
    else:
        train()
        top_1, top_5, top_10, MAP, MRR = evaluation(tag='test')
    print(f'Top 1: {top_1:.2f}')
    print(f'Top 5: {top_5:.2f}')
    print(f'Top 10: {top_10:.2f}')

    print(f'MAP: {MAP:.2f}')
    print(f'MRR: {MRR:.2f}')
    
