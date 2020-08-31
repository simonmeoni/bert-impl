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
# For the architecture, the BERT paper referenced to the original implementation of the multi-layer bidirectional
# Transformer encoder described in [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf).
# The Bert model has only one encoders stack.
# So for this part, I am using the architecture described by the paper above.
#
# ![architecture](./resources/img/model-architecture.png)

# ### Requirements

# + pycharm={"name": "#%%\n"}
import math
import torch
import copy
from torch import nn


# -

# ### Bert Encoder Stacks
# * Bert takes as input a sequence of plain text tokens
# * the output is a representation vector of the size of the hidden layers
# * Bert is a stack of multi-layer bidirectional Transformer encoder

# + pycharm={"name": "#%%\n"}
class Bert(nn.Module):
    def __init__(self, encoder, stack_size=6):
        super(Bert, self).__init__()
        self.encoderLayer = nn.ModuleList()
        for i in range(stack_size):
            self.encoderLayer.append(copy.deepcopy(encoder))
            
    def forward(self, tokens):
        representation = self.encoderLayer[0](tokens)
        for encoder in self.encoderLayer:
            representation = encoder(representation)
        return representation


# -

# ### Encoder
#
# * The encoder is composed of two modules. The first is the attention module and the second is the feed-forward network
# module.
#
# * this model is execute sequentially but the computation of each token
# is independent and could be compute concurrently

# + pycharm={"name": "#%%\n"}
class Encoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.mh_att = copy.deepcopy(MultiHeadAttention(2))
        self.feed_forward_network = nn.Linear(hidden_size, output_size)

    def forward(self, tokens):
        r = self.mh_att(tokens)
        return self.ffnn(r)


# -

# ### Self Attention
# ![attention](./resources/img/multihead-attention.png)

# + pycharm={"name": "#%%\n"}
class MultiHeadAttention(nn.Module):
    def __init__(self, att_head_number):
        super(MultiHeadAttention, self).__init__()
        self.Wo = None
        self.attentionHeads = nn.ModuleList()
        for i in range(att_head_number):
            self.attentionHeads.append(Attention())

    def forward(self, tokens):
        z_n = []
        for h in self.attentionHeads:
            z_n.append(h(tokens))
        cat_zn = torch.cat(z_n)
        return cat_zn * self.Wo


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.emb = nn.Embedding
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

# ## Pre-Training & Fine-Tuning

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Experimentation
