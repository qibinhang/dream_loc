import pandas as pd
import sys
import javalang
import multiprocessing
from utils import load_code_corpus, save_file, load_file, tokenize, split_camelcase, normalize, filter_words
from configs import Configs
from tqdm import tqdm


def extract_main_nodes_multiprocess():
    codes = []
    for _, item in codes_df.iterrows():
        codes.append((item['commit'], item['path'], item['content']))

    NUM_CPU = 10
    with multiprocessing.Pool(NUM_CPU) as p:
        results = list(tqdm(
            p.imap(_extract_main_nodes, codes), total=len(codes),
            ncols=100
        ))

    src_main_node_line = dict(results)
    save_file(configs.code_main_node_lines_path, src_main_node_line)


def extract_main_nodes():
    src_main_node_line = {}
    for _, item in tqdm(codes_df.iterrows(), desc='extracting', total=codes_df.shape[0], ncols=100):
        commit, path, content = item['commit'], item['path'], item['content']
        lines = set()
        tree = javalang.parse.parse(content)

        for _, node in tree:
            if node.position is None:
                continue
            if isinstance(node, javalang.tree.MethodReference):
                lines.add(node.position[0])
            elif isinstance(node, javalang.tree.MethodInvocation):
                lines.add(node.position[0])
            elif isinstance(node, javalang.tree.Declaration):
                lines.add(node.position[0])

        if len(lines) == 0:
            lines.add(0)
        src_main_node_line[f'{commit}/{path}'] = lines
        save_file(configs.code_main_node_lines_path, src_main_node_line)


def _extract_main_nodes(code_info):
    commit, path, content = code_info
    lines = set()
    try:
        tree = javalang.parse.parse(content)
    except:
        return f'{commit}/{path}', set(range(len(content.split('\n'))))

    for _, node in tree:
        if node.position is None:
            continue
        if isinstance(node, javalang.tree.MethodReference):
            lines.add(node.position[0])
        elif isinstance(node, javalang.tree.MethodInvocation):
            lines.add(node.position[0])
        elif isinstance(node, javalang.tree.Declaration):
            lines.add(node.position[0])
    if len(lines) == 0:
        lines.add(0)
    return f'{commit}/{path}', lines


def preprocess(sentence):
    tokens = tokenize(sentence)
    split_tokens, camel_word_split_record = split_camelcase(tokens)
    normalized_tokens = normalize(split_tokens)
    filter_tokens = filter_words(normalized_tokens)
    return filter_tokens


def generate_code_corpus(node_lines):
    contents = []
    for _, item in tqdm(codes_df.iterrows(), desc='code corpus', total=codes_df.shape[0], ncols=100):
        commit, path, content = item['commit'], item['path'], item['content']
        lines = node_lines[f'{commit}/{path}']
        content_split = content.split('\n')
        main_content = [content_split[line-1] for line in lines]
        main_content = ' '.join(main_content)
        tokens = preprocess(main_content)
        contents.append(' '.join(tokens))
    corpus = codes_df
    corpus['content'] = contents
    corpus.to_csv(configs.code_corpus_path, index=False)


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    codes_df = pd.read_csv(f'{configs.collected_code_dir}/collected_codes.csv')
    codes_df.fillna('', inplace=True)

    extract_main_nodes_multiprocess()

    main_node_lines = load_file(configs.code_main_node_lines_path)
    generate_code_corpus(main_node_lines)
