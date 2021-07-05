import torch
import torch.nn as nn
import sys
sys.path.append('..')
from Model.deep import RelevanceMatchingModel
from Model.wide import IRFeatureFusion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DreamLoc(nn.Module):
    def __init__(self, dream_dim_fusion, rmm_dim_dense, report_embedding, report_word_embedding, report_word_idf_embedding,
                 code_embedding, code_word_embedding, code_snippet_embedding, code_snippet_len_embedding,
                 code_len_embedding, k_max_pool, matrix_bugidx2path_idx2valid_path_idx,
                 irff_dim_dense, report_idx2cf, report_idx2ff, report_idx2fr, report_idx2sim, report_idx2tr,
                 report_idx2cc, with_bias, with_dropout, margin):
        super(DreamLoc, self).__init__()
        self.deep = RelevanceMatchingModel(rmm_dim_dense, report_embedding, report_word_embedding,
                                           report_word_idf_embedding, code_embedding, code_word_embedding,
                                           code_snippet_embedding, code_snippet_len_embedding,
                                           code_len_embedding, k_max_pool, matrix_bugidx2path_idx2valid_path_idx)

        self.wide = IRFeatureFusion(irff_dim_dense, report_idx2cf, report_idx2ff, report_idx2fr,
                                    report_idx2sim, report_idx2tr, report_idx2cc)

        self.fusion = nn.Sequential(
            nn.Linear(irff_dim_dense + 1, dream_dim_fusion),
            nn.Tanh(),
            nn.Linear(dream_dim_fusion, int(dream_dim_fusion / 2)),
            nn.Tanh(),
            nn.Linear(int(dream_dim_fusion / 2), 1)
        )

    def forward(self, r_idx, pos_idx, neg_idx=None):
        if neg_idx is not None:
            pos_pred = self.predict(r_idx, pos_idx)
            neg_pred = self.predict(r_idx, neg_idx)
            loss = (1 - pos_pred + neg_pred).clamp(min=1e-6)
            return loss
        else:
            pos_pred = self.predict(r_idx, pos_idx)
            return pos_pred

    def predict(self, r_idx, c_idx):
        ir_fusion = self.wide(r_idx, c_idx)
        rmm_score = self.deep(r_idx, c_idx)
        features = torch.cat([ir_fusion, rmm_score.unsqueeze(1)], dim=1)
        final_score = self.fusion(features)
        return final_score
