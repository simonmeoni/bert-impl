# ---
# jupyter:
#   jupytext:
#     formats: //notebook//ipynb,//src/main//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.7.9 64-bit (''bert'': conda)'
#     language: python
#     name: python_defaultSpec_1601643765038
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # BERT: Pre-Training of Bidirectional Tranformers for Language Understanding
# **see the full paper [here](https://arxiv.org/pdf/1810.04805.pdf)**
# -

# ## Architecture
# For the architecture, the BERT paper referenced to the original
# implementation of the multi-layer bidirectional
# Transformer encoder described in
# [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf).
# The Bert model has only one encoders stack.
# So for this part, I am using the architecture described by the paper above.
#
# ![architecture](https://tinyurl.com/y5ck5j7c)

# ### Requirements

# + pycharm={"name": "#%%\n"}
import copy
import math
import torch

from torch import nn
from torch.nn.parameter import Parameter


# -

# ### Bert Encoder Stacks
# * Bert takes as input a sequence of plain text tokens
# * the output is a representation vector of the size of the hidden layers
# * Bert is a stack of multi-layer bidirectional Transformer encoder

# + pycharm={"name": "#%%\n"}
class Bert(nn.Module):
    def __init__(self, stack_size, embedding_dim, num_embeddings, encoder):
        super().__init__()
        self.emb = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings
        )
        self.encoder_layer = nn.ModuleList()
        self.pos_enc = PositionalEncoding(embedding_dim)
        for _ in range(stack_size):
            self.encoderLayer.append(copy.deepcopy(encoder))

    def forward(self, tokens):
        cat_tokens = torch.cat(self.emb(tokens), self.pos_enc)
        representation = self.encoderLayer[0](cat_tokens)
        for encoder in self.encoderLayer:
            representation = encoder(representation)
        return representation


# -

# ### Encoder
#
# * The encoder is composed of two modules. The first is the attention module
#  and the second is the feed-forward network
# module.
#
# * this model is execute sequentially but the computation of each token
# is independent and could be compute concurrently

# + pycharm={"name": "#%%\n"}
class Encoder(nn.Module):
    def __init__(self, hidden_size, dim_w_matrices, multi_head_size, tokens_size):
        super().__init__()
        self.mh_att = MultiHeadAttention(multi_head_size, tokens_size, dim_w_matrices)
        self.add_norm_l1 = AddNormalizeLayer(dim_w_matrices)
        self.feed_forward_network = nn.Linear(hidden_size, dim_w_matrices)
        self.add_norm_l2 = AddNormalizeLayer(dim_w_matrices)

    def forward(self, embeddings):
        representations = self.mh_att(embeddings)
        return self.ffnn(representations)

# -

# ### Self Attention
# ![attention](https://tinyurl.com/y47nyfeg)

# + pycharm={"name": "#%%\n"}


def init_weights(x_n, y_n):
    return nn.init.xavier_uniform(torch.empty(x_n, y_n))


class MultiHeadAttention(nn.Module):
    def __init__(self, multi_head_size, tokens_size, dim_w_matrices):
        super().__init__()
        self.w_o = Parameter(init_weights(dim_w_matrices, tokens_size))
        self.att_heads = nn.ModuleList()
        for _ in range(multi_head_size):
            self.att_heads.append(Attention(tokens_size,dim_w_matrices))

    def forward(self, tokens):
        z_n = []
        for head in self.att_heads:
            z_n.append(head(tokens))
        return torch.cat(z_n) * self.w_o


class Attention(nn.Module):
    def __init__(self, tokens_size, dim_w_matrices):
        super().__init__()
        self.tokens_size = tokens_size
        self.dim_w_matrices = dim_w_matrices

        self.w_query = Parameter(init_weights(self.tokens_size, self.dim_w_matrices))
        self.w_key = Parameter(init_weights(self.tokens_size, self.dim_w_matrices))
        self.w_vector = Parameter(init_weights(self.tokens_size, self.dim_w_matrices))

    def forward(self, tokens):
        query = self.w_query * tokens
        key = self.w_key * tokens
        value = self.w_vector * tokens
        return nn.Softmax((query * key.t()) / math.sqrt(self.dim_w_matrices) * value)


# -

# ## Add & Normalize Layer

# + pycharm={"name": "#%%\n"}

class AddNormalizeLayer(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x_n, z_n):
        cat = torch.cat(x_n, z_n)
        return self.layer_norm(cat)
# -

# ## Positional Encoding

# + pycharm={"name": "#%%\n"}

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim

    def forward(self, pos):
        return torch.sin(pos/1000^(2/self.model_dim))
# -

# ## Pre-Training & Fine-Tuning

# ### Task #1 : Masked LM
#
#

# ### Task #2 : Next sentence Prediction

# ### Fine Tuning

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Experimentation
