from collections import OrderedDict


class Evaluator(object):
    def __init__(self):
        self.buggy_code_paths = []

    def rank(self, formatted_predict):
        pred_results = [each.pred for each in formatted_predict]
        self.buggy_code_paths = [each.buggy_code_paths for each in formatted_predict]

        ranked_result = []
        for each_report_pred_result in pred_results:
            each_ranked_result = list(sorted(each_report_pred_result, key=lambda x: x[1], reverse=True))
            ranked_result.append(OrderedDict([
                (path, (rank + 1, value)) for rank, (path, value) in enumerate(each_ranked_result)
                ]))
        return ranked_result

    def evaluate(self, ranked_result):
        hit_k = self.cal_hit_k(ranked_result)
        mean_ap = self.cal_map(ranked_result)
        mean_rr = self.cal_mrr(ranked_result)
        return hit_k, mean_ap, mean_rr

    def cal_hit_k(self, ranked_result, K=10):
        at_k = [0] * K
        num_report = len(ranked_result)

        for i, rank_info in enumerate(ranked_result):
            buggy_code_paths = self.buggy_code_paths[i]
            top_rank = min([rank_info[path][0] for path in buggy_code_paths])
            if top_rank <= K:
                at_k[top_rank - 1] += 1

        hit_k = [sum(at_k[:i+1]) / num_report for i in range(K)]
        return hit_k

    def cal_map(self, ranked_result):
        """Mean Average Precision"""
        avg_p = []
        for i, rank_info in enumerate(ranked_result):
            buggy_code_paths = self.buggy_code_paths[i]
            buggy_code_ranks = list(sorted([rank_info[path][0] for path in buggy_code_paths]))
            precision_k = [(i+1) / rank for i, rank in enumerate(buggy_code_ranks)]
            avg_p.append(sum(precision_k) / len(buggy_code_ranks))
        mean_avg_p = sum(avg_p) / len(ranked_result)
        return mean_avg_p

    def cal_mrr(self, ranked_result):
        """Mean Reciprocal Rank"""
        reciprocal_rank = []
        for i, rank_info in enumerate(ranked_result):
            buggy_code_paths = self.buggy_code_paths[i]
            top_rank = min([rank_info[path][0] for path in buggy_code_paths])
            reciprocal_rank.append(1 / top_rank)
        mrr = sum(reciprocal_rank) / len(ranked_result)
        return mrr
