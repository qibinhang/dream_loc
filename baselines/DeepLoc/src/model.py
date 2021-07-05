import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLoc(nn.Module):
    def __init__(self, summary_emb, desc_emb, code_idx2line_idx, line_idx2vec, word_emb, n_kernels):
        super(DeepLoc, self).__init__()
        self.summary_emb = nn.Embedding(summary_emb.shape[0], summary_emb.shape[1])
        self.summary_emb.weight.data.copy_(torch.from_numpy(summary_emb))
        self.summary_emb.weight.requires_grad = False

        self.desc_emb = nn.Embedding(desc_emb.shape[0], desc_emb.shape[1])
        self.desc_emb.weight.data.copy_(torch.from_numpy(desc_emb))
        self.desc_emb.weight.requires_grad = False

        self.code_idx2line_idx = nn.Embedding(code_idx2line_idx.shape[0], code_idx2line_idx.shape[1])
        self.code_idx2line_idx.weight.data.copy_(torch.from_numpy(code_idx2line_idx))
        self.code_idx2line_idx.weight.requires_grad = False

        self.line_idx2vec = nn.Embedding(line_idx2vec.shape[0], line_idx2vec.shape[1])
        self.line_idx2vec.weight.data.copy_(torch.from_numpy(line_idx2vec))
        self.line_idx2vec.weight.requires_grad = False

        self.word_emb = nn.Embedding(word_emb.shape[0], word_emb.shape[1], padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
        self.word_emb.weight.requires_grad = False

        embedding_dim = word_emb.shape[1]
        self.embedding_dim = embedding_dim
        num_kernels = n_kernels
        self.r_cnn_1 = nn.Conv2d(1, num_kernels, (3, embedding_dim), padding=(2, 0))
        self.r_cnn_2 = nn.Conv2d(1, num_kernels, (4, embedding_dim), padding=(3, 0))
        self.r_cnn_3 = nn.Conv2d(1, num_kernels, (5, embedding_dim), padding=(4, 0))

        self.c_cnn_1 = nn.Conv2d(1, num_kernels, (3, embedding_dim), padding=(2, 0))
        self.c_cnn_2 = nn.Conv2d(1, num_kernels, (4, embedding_dim), padding=(3, 0))
        self.c_cnn_3 = nn.Conv2d(1, num_kernels, (5, embedding_dim), padding=(4, 0))

        self.fusion_text = nn.Linear(num_kernels * 6, 50)
        self.fusion_text_2 = nn.Linear(50, 1)
        self.fusion_final = nn.Linear(3, 2)

    def forward(self, r_idx, c_idx, rec, fre):
        # report
        r_summary_word_indices = self.summary_emb(r_idx).long()
        r_summary = self.word_emb(r_summary_word_indices)

        r_desc = self.desc_emb(r_idx)
        r_desc = r_desc.reshape(r_idx.shape[0], -1, self.embedding_dim)

        r_vec = torch.cat([r_summary, r_desc], dim=1)
        r_vec = r_vec.unsqueeze(dim=1)

        r_out_1 = F.relu(self.r_cnn_1(r_vec)).squeeze(3)
        r_out_1 = F.max_pool1d(r_out_1, r_out_1.shape[2]).squeeze(2)

        r_out_2 = F.relu(self.r_cnn_2(r_vec)).squeeze(3)
        r_out_2 = F.max_pool1d(r_out_2, r_out_2.shape[2]).squeeze(2)

        r_out_3 = F.relu(self.r_cnn_3(r_vec)).squeeze(3)
        r_out_3 = F.max_pool1d(r_out_3, r_out_3.shape[2]).squeeze(2)

        # code
        c_line_indices = self.code_idx2line_idx(c_idx).long()
        c_vec = self.line_idx2vec(c_line_indices)
        c_vec = c_vec.unsqueeze(dim=1)

        c_out_1 = F.relu(self.c_cnn_1(c_vec)).squeeze(3)
        c_out_1 = F.max_pool1d(c_out_1, c_out_1.shape[2]).squeeze(2)

        c_out_2 = F.relu(self.c_cnn_2(c_vec)).squeeze(3)
        c_out_2 = F.max_pool1d(c_out_2, c_out_2.shape[2]).squeeze(2)

        c_out_3 = F.relu(self.c_cnn_3(c_vec)).squeeze(3)
        c_out_3 = F.max_pool1d(c_out_3, c_out_3.shape[2]).squeeze(2)

        # fusion
        out = torch.cat([r_out_1, r_out_2, r_out_3, c_out_1, c_out_2, c_out_3], dim=1)
        out = self.fusion_text(out)
        out = self.fusion_text_2(out)

        # final
        out = torch.cat([out, rec.unsqueeze(1), fre.unsqueeze(1)], dim=1)
        out = self.fusion_final(out)
        return out
