import sys
import pandas as pd
import javalang
from configs import Configs
from utils import save_file, check_dir
from tqdm import tqdm


def extract_api_info():
    src_api = {}
    codes_df = pd.read_csv(f'{configs.collected_code_dir}/collected_codes.csv')
    for _, item in tqdm(codes_df.iterrows(), desc='extracting', total=codes_df.shape[0], ncols=100):
        commit, path, content = item['commit'], item['path'], item['content']
        api = []
        try:
            tree = javalang.parse.parse(content)
            for _, node in tree.filter(javalang.tree.VariableDeclaration):
                api.append(node.type.name)
            for _, node in tree.filter(javalang.tree.FieldDeclaration):
                api.append(node.type.name)
        except:
            pass
        src_api[f'{commit}/{path}'] = api
    check_dir(configs.src_api_path)
    save_file(configs.src_api_path, src_api)


if __name__ == '__main__':
    project_name = sys.argv[1]
    configs = Configs(project_name)
    extract_api_info()
