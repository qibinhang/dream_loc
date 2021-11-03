import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class RelevanceMatchingModel(nn.Module):
    def __init__(self, dim_dense, report_embedding, report_word_embedding, report_word_idf_embedding,
                 code_embedding, code_word_embedding, code_snippet_embedding, code_snippet_len_embedding,
                 code_len_embedding, k_max_pool, matrix_bugidx2path_idx2valid_path_idx):
        super(RelevanceMatchingModel, self).__init__()
        self.matrix_bugidx2path_idx2valid_path_idx = torch.nn.Parameter(
            torch.FloatTensor(matrix_bugidx2path_idx2valid_path_idx), requires_grad=False
        )

        self.report_embedding = nn.Embedding(report_embedding.shape[0], report_embedding.shape[1])
        self.report_embedding.weight.data.copy_(torch.from_numpy(report_embedding))
        self.report_embedding.weight.requires_grad = False

        self.report_word_embedding = nn.Embedding(report_word_embedding.shape[0], report_word_embedding.shape[1])
        self.report_word_embedding.weight.data.copy_(torch.from_numpy(report_word_embedding))
        self.report_word_embedding.weight.requires_grad = False

        self.report_word_idf_embedding = nn.Embedding(report_word_idf_embedding.shape[0],
                                                      report_word_idf_embedding.shape[1])
        self.report_word_idf_embedding.weight.data.copy_(torch.from_numpy(report_word_idf_embedding))
        self.report_word_idf_embedding.weight.requires_grad = False

        self.code_embedding = nn.Embedding(code_embedding.shape[0], code_embedding.shape[1])
        self.code_embedding.weight.data.copy_(torch.from_numpy(code_embedding))
        self.code_embedding.weight.requires_grad = False

        self.code_snippet_embedding = nn.Embedding(code_snippet_embedding.shape[0], code_snippet_embedding.shape[1])
        self.code_snippet_embedding.weight.data.copy_(torch.from_numpy(code_snippet_embedding))
        self.code_snippet_embedding.weight.requires_grad = False

        self.code_word_embedding = nn.Embedding(code_word_embedding.shape[0], code_word_embedding.shape[1])
        self.code_word_embedding.weight.data.copy_(torch.from_numpy(code_word_embedding))
        self.code_word_embedding.weight.requires_grad = False

        self.code_snippet_len_embedding = nn.Embedding(code_snippet_len_embedding.shape[0],
                                                       code_snippet_len_embedding.shape[1])
        self.code_snippet_len_embedding.weight.data.copy_(torch.from_numpy(code_snippet_len_embedding))
        self.code_snippet_len_embedding.weight.requires_grad = False

        self.code_len_embedding = nn.Embedding(code_len_embedding.shape[0], code_len_embedding.shape[1])
        self.code_len_embedding.weight.data.copy_(torch.from_numpy(code_len_embedding))
        self.code_len_embedding.weight.requires_grad = False

        self.k = k_max_pool
        self.attn = DotAttn()

        self.relevance_dense = nn.Sequential(
            nn.Linear(code_word_embedding.shape[1], dim_dense),
            nn.Tanh(),
            nn.Linear(dim_dense, int(dim_dense / 2)),
            nn.Tanh(),
            nn.Linear(int(dim_dense / 2), 1),
            nn.Tanh(),
        )

        self.k_max_fusion = nn.Linear(self.k, 1)

    def forward(self, r_idx, c_idx):
        trans_c_idx = self.matrix_bugidx2path_idx2valid_path_idx[r_idx, c_idx]
        trans_c_idx = trans_c_idx.long()

        r_vec = self.report_embedding(r_idx)
        r_vec = r_vec.long()
        r_emb = self.report_word_embedding(r_vec)

        c_vec = self.code_embedding(trans_c_idx)  # [c_idx] -> [[m_idx]]
        c_vec = c_vec.long()

        s_len = self.code_snippet_len_embedding(c_vec)
        s_len = s_len.long()
        s_len = s_len.squeeze(-1)

        c_vec = self.code_snippet_embedding(c_vec)
        c_vec = c_vec.long()
        c_emb = self.code_word_embedding(c_vec)

        # 1.attention weighted code_emb
        s_mask = self.len2mask(s_len, c_vec.shape[2])
        attn_c_emb = self.attn(r_emb, c_emb, c_emb, s_mask)

        norm = torch.norm(attn_c_emb, 2, 3)
        norm = norm.unsqueeze(3).expand(-1, -1, -1, attn_c_emb.shape[3])
        norm_attn_c_emb = attn_c_emb / norm
        norm_attn_c_emb[norm_attn_c_emb != norm_attn_c_emb] = 0.0

        norm = torch.norm(r_emb, 2, 2)
        norm = norm.unsqueeze(2).expand(-1, -1, r_emb.shape[2])
        norm_r_emb = r_emb / norm
        norm_r_emb[norm_r_emb != norm_r_emb] = 0.0
        norm_r_emb = norm_r_emb.unsqueeze(2).expand(-1, -1, norm_attn_c_emb.shape[2], -1)

        # 2.a Hadamard product (element-wise multiplication) between r and attn weighted_c.
        rc_encoding = norm_r_emb * norm_attn_c_emb

        # 2.b linear fusion
        # rc_encoding = torch.tanh((self.fuse_r_attn_c(torch.cat([norm_r_emb, attn_c_emb], dim=-1))))

        # 3.relevance between r_word and code
        # r_s_relevance = torch.tanh((self.dense_1(rc_encoding)))
        # r_s_relevance = torch.tanh((self.dense_2(r_s_relevance)))
        # r_s_relevance = torch.tanh((self.dense_3(r_s_relevance))).squeeze(-1)  # (b, r_len, n_snippet)
        r_s_relevance = self.relevance_dense(rc_encoding).squeeze(-1)

        # a) pool
        # mask padded snippets
        c_len = self.code_len_embedding(c_idx).long().squeeze(-1)
        c_mask = self.len2mask(c_len, c_vec.shape[1])
        c_mask = c_mask.unsqueeze(1).expand(-1, r_s_relevance.shape[1], -1)
        r_s_relevance = r_s_relevance * c_mask

        k_max_pool = self.k_max_pooling(r_s_relevance, -1, self.k)
        # sum_pool = torch.sum(r_s_relevance, -1, keepdim=True)
        # fuse_pool = torch.cat([k_max_pool, sum_pool], dim=-1)
        # r_c_relevance = self.dense_4(k_max_pool).squeeze(-1)
        r_c_relevance = self.k_max_fusion(k_max_pool).squeeze(-1)

        # 4.report idf weighted relevance
        r_idf = self.report_word_idf_embedding(r_vec).squeeze(-1)
        mask = r_idf == 0.0
        r_idf = r_idf.masked_fill(mask, -1e12)
        r_idf = torch.softmax(r_idf, dim=1)
        weight_relevance = r_c_relevance * r_idf
        relevance_score = torch.sum(weight_relevance, dim=1)
        return relevance_score

    @staticmethod
    def len2mask(length, max_len=None):
        # assert len(length.shape) == 2
        max_len = max_len if max_len else length.max().item()
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(*length.shape, max_len)
        mask = mask < length.unsqueeze(-1)
        return mask

    @staticmethod
    def k_max_pooling(x, dim, k):
        top_k = x.topk(k, dim=dim)[0]
        return top_k


class DotAttn(nn.Module):
    def __init__(self):
        super(DotAttn, self).__init__()

    def forward(self, q, k, v, mask):
        """
        q: (b, q_len, e_dim)
        k, v: (b, num_snippet, len_snippet, e_dim)
        mask: (b, num_snippet, len_snippet)
        """
        trans_k = k.permute(0, 3, 1, 2)
        #  (b, e_dim, num_snippet x len_snippet)
        reshape_trans_k = trans_k.reshape(trans_k.shape[0], trans_k.shape[1], -1)
        attn = torch.bmm(q, reshape_trans_k)  # (b, q_len, num_snippet x len_snippet)
        # (b, q_len, num_snippet, len_snippet)
        attn = attn.reshape(attn.shape[0], attn.shape[1], trans_k.shape[2], trans_k.shape[3])
        mask = ~mask * 1e12
        mask = mask.unsqueeze(1).expand(-1, attn.shape[1], -1, -1)
        attn = attn - mask
        attn = torch.softmax(attn, dim=3)
        v = v.unsqueeze(1).expand(-1, attn.shape[1], -1, -1, -1)
        attn = attn.unsqueeze(3)
        attn_v = torch.matmul(attn, v).squeeze(3)
        return attn_v
