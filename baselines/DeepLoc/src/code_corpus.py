import pandas as pd
import sys
from configs import Configs
from tqdm import tqdm
from utils import tokenize, normalize, filter_words, split_camelcase, load_file, check_dir, save_file


def generate_code_corpus():
    corpus = [[] for _ in range(len(used_commit_path))]
    count_check = 0
    for _, item in tqdm(codes_df.iterrows(), desc='code corpus', total=codes_df.shape[0], ncols=100):
        commit, path, content = item['commit'], item['path'], item['content']
        if f'{commit}/{path}' in used_commit_path:
            content = preprocess_code_content(content)
            code_idx = used_commit_path2idx[f'{commit}/{path}']
            corpus[code_idx] = (commit, path, content)
            count_check += 1
    assert count_check == len(used_commit_path)
    columns = ['commit', 'path', 'content']
    df = pd.DataFrame(data=corpus, columns=columns)
    df.to_csv(configs.code_corpus_path, index=False)


def preprocess_code_content(code):
    preprocessed_code = []
    code_lines = code.strip().split('\n')
    package_line_num = 0
    last_import_line_num = 0
    for i, line in enumerate(code_lines):
        if line.startswith('import '):
            last_import_line_num = i
        elif line.startswith('package '):
            package_line_num = i
        elif last_import_line_num != 0 and line:
            break
    new_start_line_num = max(last_import_line_num + 1, package_line_num + 1)
    code_lines = code_lines[new_start_line_num:]

    for each_line in code_lines:
        tokens = _preprocess(each_line)
        if tokens:
            preprocessed_code.append(' '.join(tokens))
    return '\n'.join(preprocessed_code)


def _preprocess(sentence):
    tokens = tokenize(sentence)
    split_tokens, _ = split_camelcase(tokens)
    normalized_tokens = normalize(split_tokens)
    filter_tokens = filter_words(normalized_tokens)
    return filter_tokens


def get_total_commit_path():
    commit_path2idx = load_file(f'../../DeepLocator/data/{project_name}/vocabulary/commit_path2code_idx.pkl')
    code_idx2commit_path = dict(zip(commit_path2idx.values(), commit_path2idx.keys()))

    used_commit_path = set()
    for tag in ('train', 'val', 'test'):
        deep_locator_dataset = load_file(f'../../DeepLocator/data/{project_name}/dataset/{tag}_dataset_neg_200.pkl')
        for each_bug in deep_locator_dataset:
            for each_data in each_bug:
                code_idx = each_data[1]
                commit_path = code_idx2commit_path[code_idx]
                used_commit_path.add(commit_path)

    used_commit_path = list(used_commit_path)
    used_commit_path2idx = dict(zip(used_commit_path, range(len(used_commit_path))))
    new_train_dataset = update_dataset('train', code_idx2commit_path, used_commit_path2idx)
    new_val_dataset = update_dataset('val', code_idx2commit_path, used_commit_path2idx)
    new_test_dataset = update_dataset('test', code_idx2commit_path, used_commit_path2idx)
    return used_commit_path, used_commit_path2idx, new_train_dataset, new_val_dataset, new_test_dataset


def update_dataset(tag, code_idx2commit_path, used_commit_path2idx):
    dataset = []
    deep_locator_dataset = load_file(f'../../DeepLocator/data/{project_name}/dataset/{tag}_dataset_neg_200.pkl')
    for each_bug in deep_locator_dataset:
        new_each_bug = []
        for each_data in each_bug:
            new_code_idx = used_commit_path2idx[code_idx2commit_path[each_data[1]]]
            new_each_data = (each_data[0], new_code_idx, each_data[2], each_data[3], each_data[4])
            new_each_bug.append(new_each_data)
        dataset.append(new_each_bug)
    return dataset


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    codes_df = pd.read_csv(f'{configs.collected_code_dir}/collected_codes.csv')
    codes_df.fillna('', inplace=True)

    used_commit_path, used_commit_path2idx, train_dataset, val_dataset, test_dataset = get_total_commit_path()
    check_dir(configs.dataset_dir)
    save_file(f'{configs.dataset_dir}/train_dataset_neg_{configs.num_neg}.pkl', train_dataset)
    save_file(f'{configs.dataset_dir}/val_dataset_neg_{configs.num_neg}.pkl', val_dataset)
    save_file(f'{configs.dataset_dir}/test_dataset_neg_{configs.num_neg}.pkl', test_dataset)

    generate_code_corpus()
