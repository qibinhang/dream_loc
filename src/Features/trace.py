import os
import _pickle as pickle


class Trace(object):
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir

    def collect(self, report_corpus, commit2code_paths):
        trace = {}
        commit2name2paths = {}
        for commit in commit2code_paths:
            name2paths = {}
            for path in commit2code_paths[commit]:
                name = os.path.basename(path)
                paths = name2paths.get(name, [])
                paths.append(path)
                name2paths[name] = paths
            commit2name2paths[commit] = name2paths

        for report in report_corpus.itertuples():
            name2paths = commit2name2paths[report.commit]
            name_in_trace = report.trace.split('\n') if report.trace else []
            trace_paths = []
            for name in name_in_trace:
                paths = name2paths.get(name, [])
                trace_paths += paths
            trace[report.bug_id] = trace_paths
        self.save(trace)

    def save(self, trace):
        with open(f'{self.feature_dir}/trace.pkl', 'wb') as f:
            pickle.dump(trace, f)

    def load(self):
        with open(f'{self.feature_dir}/trace.pkl', 'rb') as f:
            trace = pickle.load(f)
        return trace
