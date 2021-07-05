import git
import os
import pandas as pd
import sys
import xml.etree.cElementTree as ET
import _pickle as pickle
from glob import glob
from os.path import exists
from tqdm import tqdm
sys.path.append('..')
from configures import Configs
from utils import sort_reports_according_to_commit_timestamp


class CodeCollector(object):
    """collect 'ADD' and 'Modify' codes in each commit version"""

    def __init__(self, report_path, code_dir, corpus_dir):
        self.report_path = report_path
        self.code_dir = code_dir
        self.corpus_dir = corpus_dir
        self.collected_codes_dir = f'{corpus_dir}/collected_codes'
        if not exists(self.collected_codes_dir):
            os.mkdir(self.collected_codes_dir)

    def collect_codes_at_each_commit(self):
        root = ET.parse(self.report_path).getroot()
        reports = list(root.iter('table'))
        # SWT: 247e8fc, tomcat: 42056be, birt: 345f01b, eclipse: 5da5952, jdt: 8d013d0
        repo = git.Repo(self.code_dir)  # 确保当前git的HEAD在最新的缺陷报告的commit!
        assert repo.bare is False
        repo = repo.git

        git_log = repo.log('--reverse', '--oneline')
        commit_order = [each_line.split()[0] for each_line in git_log.split('\n')]
        reports = sort_reports_according_to_commit_timestamp(reports, commit_order,
                                                             save_path=f'{self.report_path[:-4]}_sorted.pkl')

        collected_codes = []
        commit_order = []
        commit2code_paths = {}
        for i, r in enumerate(tqdm(reports, desc='collecting codes')):
            commit = r[7].text
            commit_order.append(commit)
            repo.checkout(commit + '~1', '-f')
            code_paths = glob(f'{self.code_dir}/**/*.java', recursive=True)
            # NOTE:
            # here are some code paths in commit2code_paths, but these code files are not actually collected.
            # collect 'A', 'M', 'Rxxx'。
            commit2code_paths[commit] = [os.path.relpath(p, start=self.code_dir) for p in code_paths]

            if i != 0:
                last_commit = commit_order[i - 1]
                diff_info = repo.diff("--name-status", last_commit + '~1', commit + '~1', "--", "*.java")  # 可能为空
                add_mod_code_paths = self.get_add_mod_code_paths(diff_info)
                if not add_mod_code_paths:
                    continue
                add_mod_code_paths = [f'{self.code_dir}/{p}' for p in add_mod_code_paths]
            else:  # first commit as init version
                add_mod_code_paths = code_paths

            # codes come from 'commit ~1' but dir name is 'commit' for convenience
            add_mod_codes = self.collect_codes(add_mod_code_paths)
            assert len(add_mod_codes) == len(add_mod_code_paths)
            add_mod_code_paths = [os.path.relpath(p, start=self.code_dir) for p in add_mod_code_paths]
            collected_codes += list(zip([commit]*len(add_mod_code_paths), add_mod_code_paths, add_mod_codes))

        columns = ['commit', 'path', 'content']
        df = pd.DataFrame(data=collected_codes, columns=columns)
        df.to_csv(f'{self.collected_codes_dir}/collected_codes.csv', index=False)

        with open(f'{self.collected_codes_dir}/commit2code_paths.pkl', 'wb') as f:
            pickle.dump(commit2code_paths, f)

        with open(f'{self.collected_codes_dir}/commit_order.txt', 'w') as f:
            f.write('\n'.join(commit_order))

    @staticmethod
    def get_add_mod_code_paths(diff_info):
        """get the path of file that were “Added” or “Modified" between the commit~1 and  last commit~1"""
        add_mod_code_paths = []
        if len(diff_info) == 0:
            return add_mod_code_paths
        for each_file_info in diff_info.split('\n'):
            info = each_file_info.split('\t')
            action = info[0]
            if action in ('A', 'M'):
                add_mod_code_paths.append(info[1])
            elif action[0] == 'R':
                add_mod_code_paths.append(info[2])
            # # TEST
            # elif action[0] == 'C':
            #     print('')
            # #
        return add_mod_code_paths

    @staticmethod
    def collect_codes(paths):
        codes = []
        for p in paths:
            with open(p, 'r', errors='ignore') as f:
                content = f.read()
            codes.append(content)
        return codes


def main(project_name):
    configs = Configs(project_name)
    print(f"report_path: {configs.report_path}")
    print(f"code_dir: {configs.code_dir}")
    print(f"corpus_dir: {configs.corpus_dir}\n")

    if not exists(configs.corpus_dir):
        os.makedirs(configs.corpus_dir)
    cc = CodeCollector(configs.report_path, configs.code_dir, configs.corpus_dir)
    cc.collect_codes_at_each_commit()


if __name__ == '__main__':
    project = sys.argv[1]
    main(project)
