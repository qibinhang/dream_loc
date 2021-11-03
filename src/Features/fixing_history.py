import copy
import logging
import numpy as np
import xml.etree.cElementTree as ET
import _pickle as pickle
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


class FixingHistory:
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir

    def collect(self, report_corpus):
        reports = self.sort_reports(report_corpus)
        code_commit_timestamp = self.collect_code_commit_timestamp(reports)
        recency, frequency = {}, {}
        last_report_recency, last_report_frequency = {}, {}
        for r in reports:
            bug_id = r[0]
            buggy_paths = r[1]
            report_timestamp = r[2]
            commit = r[4]
            each_bug_recency, each_bug_frequency = self._collect_each_bug_history(report_timestamp, buggy_paths,
                                                                                  code_commit_timestamp)
            last_report_recency.update(each_bug_recency)
            last_report_frequency.update(each_bug_frequency)
            recency[bug_id] = copy.deepcopy(last_report_recency)
            frequency[bug_id] = copy.deepcopy(last_report_frequency)

        norm_recency = self.normalize(recency)
        norm_frequency = self.normalize(frequency)
        self.save(norm_recency, norm_frequency)

    @staticmethod
    def sort_reports(report_corpus):
        reports = []
        for report in report_corpus.itertuples():
            bug_id = str(report.bug_id)
            report_timestamp = int(report.report_timestamp)
            commit_timestamp = int(report.commit_timestamp)
            buggy_file_paths = set(report.buggy_paths.split('\n'))
            commit = report.commit
            reports.append((bug_id, buggy_file_paths, report_timestamp, commit_timestamp, commit))
        reports_sorted_by_time = list(sorted(reports, key=lambda x: x[2]))
        return reports_sorted_by_time

    @staticmethod
    def collect_code_commit_timestamp(reports):
        code_commit_timestamp = {}
        for r in reports:
            for path in r[1]:
                each_code_commit_timestamps = code_commit_timestamp.get(path, [])
                each_code_commit_timestamps.append(r[3])
                code_commit_timestamp[path] = each_code_commit_timestamps
        for path, each_code_commit_timestamps in code_commit_timestamp.items():
            sorted_ecct = list(sorted(each_code_commit_timestamps))
            code_commit_timestamp[path] = sorted_ecct
        return code_commit_timestamp

    @staticmethod
    def _collect_each_bug_history(report_timestamp, buggy_paths, code_commit_timestamp):
        each_bug_recency, each_bug_frequency = {}, {}
        for path in buggy_paths:
            fix_count = 0
            last_commit_timestamp = 0
            each_path_commit_timestamps = code_commit_timestamp[path]
            for commit_timestamp in each_path_commit_timestamps:
                if commit_timestamp < report_timestamp:
                    last_commit_timestamp = commit_timestamp
                    fix_count += 1
                else:
                    break
            if last_commit_timestamp != 0:
                num_month = (report_timestamp - last_commit_timestamp) // 2592000 + 1
                each_bug_recency[path] = 1 / num_month
                each_bug_frequency[path] = fix_count
        return each_bug_recency, each_bug_frequency

    def save(self, recency, frequency):
        with open(f'{self.feature_dir}/fixing_recency.pkl', 'wb') as f:
            pickle.dump(recency, f)
        with open(f'{self.feature_dir}/fixing_frequency.pkl', 'wb') as f:
            pickle.dump(frequency, f)

    def load_fixing_recency(self):
        with open(f'{self.feature_dir}/fixing_recency.pkl', 'rb') as f:
            fr = pickle.load(f)
        return fr

    def load_fixing_frequency(self):
        with open(f'{self.feature_dir}/fixing_frequency.pkl', 'rb') as f:
            ff = pickle.load(f)
        return ff

    @staticmethod
    def normalize(fixing_info):
        max_value = 0
        for bug_id in fixing_info:
            each_report_info = fixing_info[bug_id]
            if not each_report_info:
                continue
            each_max = max(each_report_info.values())
            if each_max > max_value:
                max_value = each_max

        for bug_id in fixing_info:
            each_report_ff = fixing_info[bug_id]
            for buggy_code_path in each_report_ff:
                # the frequency/recency of those non-buggy file is 0,
                # so the min_frequency is 0. (freq - 0) / (max_freq - 0)
                norm_ff = each_report_ff[buggy_code_path] / max_value
                norm_ff = 1 / (1 + np.exp(-norm_ff))
                each_report_ff[buggy_code_path] = norm_ff
        return fixing_info
