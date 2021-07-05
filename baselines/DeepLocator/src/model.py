import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLocator(nn.Module):
    def __init__(self, report_emb, code_emb, word_emb, n_kernels):
        super(DeepLocator, self).__init__()
        self.report_emb = nn.Embedding(report_emb.shape[0], report_emb.shape[1])
        self.report_emb.weight.data.copy_(torch.from_numpy(report_emb))
        self.report_emb.weight.requires_grad = False

        self.code_emb = nn.Embedding(code_emb.shape[0], code_emb.shape[1])
        self.code_emb.weight.data.copy_(torch.from_numpy(code_emb))
        self.code_emb.weight.requires_grad = False

        self.word_emb = nn.Embedding(word_emb.shape[0], word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
        self.word_emb.weight.requires_grad = False

        num_kernels = n_kernels
        self.r_cnn_0 = nn.Conv2d(1, num_kernels, (2, 100), padding=(1, 0))
        self.r_cnn_1 = nn.Conv2d(1, num_kernels, (3, 100), padding=(2, 0))
        self.r_cnn_2 = nn.Conv2d(1, num_kernels, (4, 100), padding=(3, 0))
        self.r_cnn_3 = nn.Conv2d(1, num_kernels, (5, 100), padding=(4, 0))
        self.r_text = nn.Linear(num_kernels * 3, 50)

        self.c_cnn_0 = nn.Conv2d(1, num_kernels, (2, 100), padding=(1, 0))
        self.c_cnn_1 = nn.Conv2d(1, num_kernels, (3, 100), padding=(2, 0))
        self.c_cnn_2 = nn.Conv2d(1, num_kernels, (4, 100), padding=(3, 0))
        self.c_cnn_3 = nn.Conv2d(1, num_kernels, (5, 100), padding=(4, 0))
        self.c_text = nn.Linear(num_kernels * 3, 50)

        self.fusion_text = nn.Linear(num_kernels * 8, 50)
        self.fusion_text_2 = nn.Linear(50, 1)
        self.fusion_final = nn.Linear(3, 2)

    def forward(self, r_idx, c_idx, rec, fre):
        # report
        r_word_indices = self.report_emb(r_idx)
        r_word_indices = r_word_indices.long()
        r_vec = self.word_emb(r_word_indices)
        r_vec = r_vec.unsqueeze(dim=1)

        r_out_0 = F.relu(self.r_cnn_0(r_vec)).squeeze(3)
        r_out_0 = F.max_pool1d(r_out_0, r_out_0.shape[2]).squeeze(2)

        r_out_1 = F.relu(self.r_cnn_1(r_vec)).squeeze(3)
        r_out_1 = F.max_pool1d(r_out_1, r_out_1.shape[2]).squeeze(2)

        r_out_2 = F.relu(self.r_cnn_2(r_vec)).squeeze(3)
        r_out_2 = F.max_pool1d(r_out_2, r_out_2.shape[2]).squeeze(2)

        r_out_3 = F.relu(self.r_cnn_3(r_vec)).squeeze(3)
        r_out_3 = F.max_pool1d(r_out_3, r_out_3.shape[2]).squeeze(2)
        # r_cnn_out = torch.cat([r_out_1, r_out_2, r_out_3], dim=1)
        # r_encode = self.r_text(r_cnn_out)

        # code
        c_word_indices = self.code_emb(c_idx)
        c_word_indices = c_word_indices.long()
        c_vec = self.word_emb(c_word_indices)
        c_vec = c_vec.unsqueeze(dim=1)

        c_out_0 = F.relu(self.c_cnn_0(c_vec)).squeeze(3)
        c_out_0 = F.max_pool1d(c_out_0, c_out_0.shape[2]).squeeze(2)

        c_out_1 = F.relu(self.c_cnn_1(c_vec)).squeeze(3)
        c_out_1 = F.max_pool1d(c_out_1, c_out_1.shape[2]).squeeze(2)

        c_out_2 = F.relu(self.c_cnn_2(c_vec)).squeeze(3)
        c_out_2 = F.max_pool1d(c_out_2, c_out_2.shape[2]).squeeze(2)

        c_out_3 = F.relu(self.c_cnn_3(c_vec)).squeeze(3)
        c_out_3 = F.max_pool1d(c_out_3, c_out_3.shape[2]).squeeze(2)
        # c_cnn_out = torch.cat([c_out_1, c_out_2, c_out_3], dim=1)
        # c_encode = self.c_text(c_cnn_out)

        # fusion
        out = torch.cat([r_out_0, r_out_1, r_out_2, r_out_3, c_out_0, c_out_1, c_out_2, c_out_3], dim=1)
        out = self.fusion_text(out)
        out = self.fusion_text_2(out)

        # final
        out = torch.cat([out, rec.unsqueeze(1), fre.unsqueeze(1)], dim=1)
        out = self.fusion_final(out)
        return out


# class DeepLocator(nn.Module):
#     def __init__(self, report_emb, code_emb, word_emb):
#         super(DeepLocator, self).__init__()
#         self.report_emb = nn.Embedding(report_emb.shape[0], report_emb.shape[1])
#         self.report_emb.weight.data.copy_(torch.from_numpy(report_emb))
#         self.report_emb.weight.requires_grad = False
#
#         self.code_emb = nn.Embedding(code_emb.shape[0], code_emb.shape[1])
#         self.code_emb.weight.data.copy_(torch.from_numpy(code_emb))
#         self.code_emb.weight.requires_grad = False
#
#         self.word_emb = nn.Embedding(word_emb.shape[0], word_emb.shape[1], padding_idx=0)
#         self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
#         self.word_emb.weight.requires_grad = False
#
#         num_kernels = 30
#         self.cnn_1 = nn.Conv2d(1, num_kernels, (4, 100), padding=(3, 0))
#         self.cnn_2 = nn.Conv2d(1, num_kernels, (5, 100), padding=(4, 0))
#         self.cnn_3 = nn.Conv2d(1, num_kernels, (6, 100), padding=(5, 0))
#         self.fc_text = nn.Linear(num_kernels * 3, 1)
#         self.fc_final = nn.Linear(3, 2)
#
#     def forward(self, r_idx, c_idx, rec, fre):
#         r_word_indices = self.report_emb(r_idx)
#         r_word_indices = r_word_indices.long()
#         r_vec = self.word_emb(r_word_indices)
#
#         c_word_indices = self.code_emb(c_idx)
#         c_word_indices = c_word_indices.long()
#         c_vec = self.word_emb(c_word_indices)
#
#         rc_vec = torch.cat([r_vec, c_vec], dim=1)
#         rc_vec = rc_vec.unsqueeze(dim=1)
#
#         out_1 = F.relu(self.cnn_1(rc_vec)).squeeze(3)
#         out_1 = F.max_pool1d(out_1, out_1.shape[2]).squeeze(2)
#
#         out_2 = F.relu(self.cnn_2(rc_vec)).squeeze(3)
#         out_2 = F.max_pool1d(out_2, out_2.shape[2]).squeeze(2)
#
#         out_3 = F.relu(self.cnn_3(rc_vec)).squeeze(3)
#         out_3 = F.max_pool1d(out_3, out_3.shape[2]).squeeze(2)
#
#         out = torch.cat([out_1, out_2, out_3], dim=1)
#         # out = F.dropout(out, p=0.3, training=self.training)
#         out = self.fc_text(out)
#         out = torch.cat([out, rec.unsqueeze(1).float(), fre.unsqueeze(1).float()], dim=1)
#         out = self.fc_final(out)
#         return out
