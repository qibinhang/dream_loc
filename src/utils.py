import itertools
import numpy as np
import os
import _pickle as pickle
try:
    from sklearn.externals import joblib
except:
    import joblib


def load_file(path):
    assert path[-4:] == '.pkl'
    try:
        with open(path, 'rb') as f:
            file = pickle.load(f)
    except FileNotFoundError:
        with open(f'{path[:-4]}.joblib', 'rb') as f:
            file = joblib.load(f)
    return file


def save_file(path, file):
    assert path[-4:] == '.pkl'
    try:
        with open(path, 'wb') as f:
            pickle.dump(file, f)
    except OverflowError:
        os.remove(path)
        with open(f'{path[:-4]}.joblib', 'wb') as f:
            joblib.dump(file, f)


def sort_by_bugidx(inputs):
    """
    :param inputs: [(bugidx, [score])]
    :return: [[scores]]
    """
    sorted_inputs = list(sorted(inputs, key=lambda x: x[0]))
    scores = [item[1] for item in sorted_inputs]
    return scores


def list2matrix(inputs):
    """
    :param inputs: [[scores]]
    :return: np.array, pad with 0.0
    """
    matrix = list(itertools.zip_longest(*inputs, fillvalue=0.0))
    matrix = np.array(matrix).transpose((1, 0))
    return matrix


def sort_reports_according_to_commit_timestamp(reports, commit_order, save_path=None):
    from tqdm import tqdm
    sorted_report = []
    for r in tqdm(reports, desc='sort report according to commit order'):
        r_commit = r[7].text
        r_idx = None
        for idx, commit in enumerate(commit_order):
            if commit.startswith(r_commit):
                r_idx = idx
        if r_idx is None:
            raise ValueError('请确保当前git的HEAD在最新的缺陷报告的commit。')
        sorted_report.append((r_idx, r))
    reports = list(sorted(sorted_report, key=lambda x: x[0]))
    reports = [item[1] for item in reports]
    # last_commit_timestamp = 0
    # for idx, r in enumerate(reports):
    #     commit_timestamp = int(r[8].text)
    #     if commit_timestamp < last_commit_timestamp:
    #         last_r_bugid = reports[idx-1][1].text
    #         current_r_bugid = r[1].text
    #         print(f'last report {last_r_bugid} commit_timestamp: {last_commit_timestamp}')
    #         print(f'current report {current_r_bugid} commit_timestamp: {commit_timestamp}')
    #         assert commit_timestamp >= last_commit_timestamp
    #
    #     last_commit_timestamp = commit_timestamp

    if save_path is not None:
        save_file(save_path, reports)
    return reports
