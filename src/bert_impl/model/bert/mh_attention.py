import math

from torch.nn import functional as f
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, multi_head_size, dim_model):
        super().__init__()
        self.dim_model = dim_model
        self.multi_head_size = multi_head_size
        self.linear_o = nn.Linear(self.dim_model, self.dim_model)
        self.query = nn.Linear(self.dim_model, self.dim_model)
        self.key = nn.Linear(self.dim_model, self.dim_model)
        self.value = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, tokens, mask=None):
        batch_size = tokens.shape[0]
        z_n = self.compute_attention(tokens, batch_size, mask)
        return self.linear_o(z_n.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model))

    def compute_attention(self, tokens, batch_size, mask=None):
        d_k = self.dim_model // self.multi_head_size
        query_mat = self.query(tokens).view(batch_size, -1, self.multi_head_size, d_k) \
            .transpose(2, 1)
        key_mat = self.key(tokens).view(batch_size, -1, self.multi_head_size, d_k) \
            .transpose(2, 1)
        value_mat = self.value(tokens).view(batch_size, -1, self.multi_head_size, d_k) \
            .transpose(2, 1)

        scores = (query_mat.matmul(key_mat.transpose(-2, -1)) / math.sqrt(self.dim_model))
        if mask is not None:
            scores.masked_fill(mask == 0, 1e-11)
        return f.softmax(scores, dim=-1).matmul(value_mat)
