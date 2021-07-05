import torch
import torch.nn as nn


class IRFeatureFusion(nn.Module):
    def __init__(self, dim_dense, report_idx2cf, report_idx2ff, report_idx2fr,
                 report_idx2sim, report_idx2tr, report_idx2cc):
        super(IRFeatureFusion, self).__init__()
        self.report_idx2cf = torch.nn.Parameter(torch.FloatTensor(report_idx2cf), requires_grad=False)
        self.report_idx2ff = torch.nn.Parameter(torch.FloatTensor(report_idx2ff), requires_grad=False)
        self.report_idx2fr = torch.nn.Parameter(torch.FloatTensor(report_idx2fr), requires_grad=False)
        self.report_idx2sim = torch.nn.Parameter(torch.FloatTensor(report_idx2sim), requires_grad=False)
        self.report_idx2cc = torch.nn.Parameter(torch.FloatTensor(report_idx2cc), requires_grad=False)

        self.fusion = torch.nn.Linear(5, dim_dense)

    def forward(self, r_idx, c_idx):
        cf = self.report_idx2cf[r_idx, c_idx].unsqueeze(1)
        ff = self.report_idx2ff[r_idx, c_idx].unsqueeze(1)
        fr = self.report_idx2fr[r_idx, c_idx].unsqueeze(1)
        sim = self.report_idx2sim[r_idx, c_idx].unsqueeze(1)
        cc = self.report_idx2cc[r_idx, c_idx].unsqueeze(1)
        feature_fusion = self.fusion(torch.cat([sim, cc, cf, ff, fr], dim=1))
        return feature_fusion
