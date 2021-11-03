import lizard
import multiprocessing
import os
import _pickle as pickle
from tqdm import tqdm


class CyclomaticComplexity(object):
    def __init__(self, feature_dir, num_cpu=1):
        self.feature_dir = feature_dir
        self.num_cpu = min(num_cpu, multiprocessing.cpu_count()-1)

    def calculate_cyclomatic_complexity(self, codes):
        """codes: [(path, code_content)]"""
        with multiprocessing.Pool(self.num_cpu) as p:
            results = list(tqdm(
                p.imap(self.cal_cc, codes), total=len(codes), ncols=100
            ))
        cyclomatic_complexity = dict(results)
        cyclomatic_complexity = self.normalize(cyclomatic_complexity)
        self.save(cyclomatic_complexity)

    @staticmethod
    def cal_cc(code):
        path, content = code
        name = os.path.basename(path)
        cc = lizard.analyze_file.analyze_source_code(name, content).CCN
        return path, cc

    @staticmethod
    def normalize(cyclomatic_complexity):
        max_cc, min_cc = max(list(cyclomatic_complexity.values())), min(list(cyclomatic_complexity.values()))
        for path in cyclomatic_complexity:
            cc = cyclomatic_complexity[path]
            cc = (cc - min_cc) / (max_cc - min_cc)
            cyclomatic_complexity[path] = cc
        return cyclomatic_complexity

    def save(self, cyclomatic_complexity):
        with open(f'{self.feature_dir}/cyclomatic_complexity.pkl', 'wb') as f:
            pickle.dump(cyclomatic_complexity, f)

    def load(self):
        with open(f'{self.feature_dir}/cyclomatic_complexity.pkl', 'rb') as f:
            cc = pickle.load(f)
        return cc
