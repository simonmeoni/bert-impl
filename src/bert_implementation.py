# ---
# jupyter:
#   jupytext:
#     formats: //notebook//ipynb,//src/main//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # BERT: Pre-Training of Bidrectional Tranformers for Language Understanding
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
# ![architecture](https://tinyurl.com/yye57wlx)

# ### Requirements

# + pycharm={"name": "#%%\n"}
import math
import copy
import torch
from torch import nn


# -

# ### Bert Encoder Stacks
# * Bert takes as input a sequence of plain text tokens
# * the output is a representation vector of the size of the hidden layers
# * Bert is a stack of multi-layer bidirectional Transformer encoder

# + pycharm={"name": "#%%\n"}
class Bert(nn.Module):
    def __init__(self, encoder, stack_size, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings
        )
        self.encoder_layer = nn.ModuleList()
        # self.pos_enc = PositionalEncoding()
        for _ in range(stack_size):
            self.encoderLayer.append(copy.deepcopy(encoder))

    def forward(self, tokens):
        embeddings = self.emb(tokens)
        representation = self.encoderLayer[0](embeddings)
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
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mh_att = copy.deepcopy(MultiHeadAttention(2))
        self.add_norm_l1 = AddNormalizeLayer(output_size)
        self.feed_forward_network = nn.Linear(hidden_size, output_size)
        self.add_norm_l2 = AddNormalizeLayer(output_size)

    def forward(self, embeddings):
        representations = self.mh_att(embeddings)
        return self.ffnn(representations)


# -

# ### Self Attention
# ![attention](https://tinyurl.com/y6qzem5l)

# + pycharm={"name": "#%%\n"}
class MultiHeadAttention(nn.Module):
    def __init__(self, att_head_number):
        super().__init__()
        self.w_o = None
        self.att_heads = nn.ModuleList()
        for _ in range(att_head_number):
            self.att_heads.append(Attention())

    def forward(self, tokens):
        z_n = []
        for head in self.att_heads:
            z_n.append(head(tokens))
        # cat_zn = torch.cat(z_n)
        return z_n * self.w_o


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_query = None
        self.w_key = None
        self.w_vector = None

    def forward(self, tokens):
        emb = self.emb(tokens)
        # use dot product instead
        query = self.w_query * emb
        key = self.w_key * emb
        value = self.w_vector * emb
        # use matrix form to simplify the code
        # dot product or matrix multiplication ?
        # I confused and merge multi-head attention and self-attention
        return nn.Softmax((query * key.t()) / math.sqrt(8) * value)


# -

# ## Add & Normalize Layer

# + pycharm={"name": "#%%\n"}
class AddNormalizeLayer(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x_n, z_n):
        cat = torch.cat((x_n, z_n), 0)
        return self.layer_norm(cat)
# -

# ## Pre-Training & Fine-Tuning

# ### Task #1 : Masked LM
#
#

# ### Task #2 : Next sentence Prediction

# ### Fine Tuning

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Experimentation
