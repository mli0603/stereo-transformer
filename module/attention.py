#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionRelative(nn.MultiheadAttention):
    """
    Multihead attention with relative positional encoding
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionRelative, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

    def gather_attn(self, indexes, bsz, dim, attn):
        """
        indexes [LxL]: indexes to shift attn
        attn q k_r [N,L,2L-1]: gather along dimension -1
            L: target len
            N: batch size
        attn q_r k [N,2L-1,L]: gather along dimension -2
            L: target len
            N: batch size
        """

        indexes = indexes.unsqueeze(0).expand([bsz, -1, -1])  # N x L x L
        attn = torch.gather(attn, dim, indexes)

        return attn

    def forward(self, query, key, value, need_weights=True, attn_mask=None, pos_enc=None, pos_indexes=None):
        tgt_len, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # project to get qkv
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # cross-attention
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        # project to find q_r, k_r
        if pos_enc is not None:
            # compute k_r, q_r
            _start = 0
            _end = 2 * embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            _b = self.in_proj_bias[_start:_end]
            q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # 2L-1xNxE
            if bsz == 2 * q_r.size(1):  # this is when left/right features are cat together
                q_r, k_r = torch.cat([q_r, q_r], dim=1), torch.cat([k_r, k_r], dim=1)
        else:
            q_r = None
            k_r = None

        # scale query
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        if q_r is not None:
            q_r = q_r * scaling

        # adjust attn mask size
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # reshape
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)  # N*n_head x L x E
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if q_r is not None:  # N*n_head x 2L-1 x E
            q_r = q_r.contiguous().view(2 * tgt_len - 1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k_r is not None:
            k_r = k_r.contiguous().view(2 * tgt_len - 1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        # compute attn weight
        attn_feat = torch.bmm(q, k.transpose(1, 2))  # N*n_head x L x L

        # add positional terms
        if pos_enc is not None:
            # 0.3 s
            attn_feat_pos = torch.einsum('ijk,ilk->ijl', q, k_r)  # N*n_head x L x 2L -1
            attn_feat_pos = self.gather_attn(pos_indexes, bsz * self.num_heads, -1, attn_feat_pos)
            attn_pos_feat = torch.einsum('ijk,ilk->ijl', q_r, k)  # N*n_head x 2L -1 x L
            attn_pos_feat = self.gather_attn(pos_indexes, bsz * self.num_heads, -2, attn_pos_feat)

            # 0.1 s
            attn_output_weights = attn_feat + attn_feat_pos + attn_pos_feat
        else:
            attn_output_weights = attn_feat

        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # apply attn mask
        if attn_mask is not None:
            attn_output_weights += attn_mask

        # raw attn
        raw_attn_output_weights = attn_output_weights

        # softmax
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        # compute v
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads

        # raw attn
        raw_attn_output_weights = raw_attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        raw_attn_output_weights = raw_attn_output_weights.sum(dim=1)

        return attn_output, attn_output_weights, raw_attn_output_weights
