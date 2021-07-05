import xml.etree.cElementTree as ET
import sys
import _pickle as pickle
sys.path.append('..')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        self.path2bugids = None
        self.path2commit_times = None
        self.bugid2commit_time = None

    def collect(self, report_path, report_corpus, commit2code_paths):
        # some reports are filtered because of no valid buggy files.
        valid_bug_ids = list(map(str, report_corpus['bug_id'].tolist()))
        reports = self.load_and_sort_reports(report_path, valid_bug_ids)
        self.path2bugids, self.path2commit_times = self.gen_code_path_to_bug_ids_and_commit_time(reports)
        self.bugid2commit_time = self.gen_bugid_to_commit_time(reports)
        bugid2commit = self.gen_bugid_to_commit(reports)
        bugid2related_bugids = self.gen_bugid_to_related_bugids(commit2code_paths, bugid2commit)
        bugid2summary = dict(list(zip(valid_bug_ids, report_corpus['summary'].tolist())))
        collective_filtering_score = self.gen_collaborative_score(bugid2summary, bugid2related_bugids)
        self.save(collective_filtering_score)

    @staticmethod
    def gen_code_path_to_bug_ids_and_commit_time(reports):
        """
        {buggy_path: [bugid]}
        {buggy_path: [commit_times]}
        """
        path2bugids = {}
        path2commit_times = {}
        for r in reports:
            buggy_files = r[1]
            for bf in buggy_files:
                bugids = path2bugids.get(bf, [])
                bugids.append(r[0])
                path2bugids[bf] = bugids

                commit_times = path2commit_times.get(bf, [])
                commit_times.append(r[3])
                path2commit_times[bf] = commit_times
        return path2bugids, path2commit_times

    @staticmethod
    def gen_bugid_to_commit_time(reports):
        bugid2commit_time = {}
        for r in reports:
            bugid2commit_time[r[0]] = r[3]
        return bugid2commit_time

    @staticmethod
    def gen_bugid_to_commit(reports):
        bugid2commit = {}
        for r in reports:
            bugid2commit[r[0]] = r[2]
        return bugid2commit

    def get_related_bugids(self, code_path, bugid):
        """
        :return [related_bugid]
        related_bugid: the 'code_path' is one of buggy files for a bug report,
        and the bug report's commit time is small than that of 'bugid'.
        """
        related_bugids = []
        commit_times = self.path2commit_times.get(code_path, [])
        if commit_times:
            current_r_commit_time = self.bugid2commit_time[bugid]
            idx = 0
            for i, ct in enumerate(commit_times):
                if ct >= current_r_commit_time:  # in order to filter 'bugid' itself, use '>=' instead of '>'
                    idx = i
                    break
            if idx != 0:
                bugids = self.path2bugids[code_path]
                related_bugids = bugids[:idx]
        return related_bugids

    def gen_bugid_to_related_bugids(self, commit2code_paths, bugid2commit):
        """related_dict: {bugid: {commit_path: [bugids}}"""
        related_dict = {}
        for bugid in bugid2commit:
            commit = bugid2commit[bugid]
            code_paths = commit2code_paths[commit]
            each_r_related_dict = {}
            for path in code_paths:
                related_bugids = self.get_related_bugids(path, bugid)
                if related_bugids:
                    each_r_related_dict[path] = related_bugids
            related_dict[bugid] = each_r_related_dict
        return related_dict

    @staticmethod
    def gen_collaborative_score(bugid2summary, bugid2related_bugids):
        """score: {bugid: {path: score}}"""
        score = {}
        # bugid2summary = dict([(report.bug_id, report.summary) for report in report_corpus])
        tf_idf = TfidfVectorizer().fit(list(bugid2summary.values()))
        for bugid in bugid2summary:
            related_summaries = []
            summary = bugid2summary[bugid]
            summary_vec = tf_idf.transform([summary])
            related_bugids = bugid2related_bugids[bugid]
            for path in related_bugids:
                related_summary = [bugid2summary[bugid] for bugid in related_bugids[path]
                                   if bugid in bugid2summary]
                related_summaries.append(' '.join(related_summary))
            if not related_summaries:  # process the first bugid in related_bugids
                score[str(bugid)] = {}
                continue
            related_summaries_vec = tf_idf.transform(related_summaries)
            cos_sim = cosine_similarity(summary_vec, related_summaries_vec)
            score[bugid] = dict(zip(list(related_bugids.keys()), cos_sim[0]))
        return score

    @staticmethod
    def load_and_sort_reports(report_path, valid_bug_ids):
        reports = []
        root = ET.parse(report_path).getroot()
        for table in root.iter('table'):
            bug_id = table[1].text
            if bug_id not in valid_bug_ids:
                continue
            commit = table[7].text
            commit_timestamp = int(table[8].text)
            buggy_file_paths = set(table[9].text.split('\n'))
            reports.append((bug_id, buggy_file_paths, commit, commit_timestamp))
        reports_sorted_by_time = list(sorted(reports, key=lambda x: x[3]))
        return reports_sorted_by_time

    def save(self, collective_filtering_score):
        with open(f'{self.feature_dir}/collaborative_filtering.pkl', 'wb') as f:
            pickle.dump(collective_filtering_score, f)

    def load(self):
        with open(f'{self.feature_dir}/collaborative_filtering.pkl', 'rb') as f:
            collective_filtering_score = pickle.load(f)
        return collective_filtering_score
