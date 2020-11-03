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
#     metadata:
#       interpreter:
#         hash: 52a97d57a70876463eed4fac2064bbbe8674799a9b35183dbfc475f4ebf43b46
#     name: 'Python 3.7.9 64-bit (''bert'': conda)'
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
import math
import torch
import spacy

import pandas as pd
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

# -

# ### Bert Encoder Stacks
# * Bert takes as input a sequence of plain text tokens
# * the output is a representation vector of the size of the hidden layers
# * Bert is a stack of multi-layer bidirectional Transformer encoder

# + pycharm={"name": "#%%\n"}
class Bert(nn.Module):
    # pylint: disable=too-many-arguments
    def __init__(self, stack_size, embedding_dim, num_embeddings, dim_w_matrices, mh_size):
        super().__init__()
        self.emb = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings
        )
        self.encoder_layer = nn.ModuleList()
        for _ in range(stack_size):
            self.encoder_layer.append(Encoder(dim_w_matrices, mh_size, embedding_dim))

    def forward(self, tokens):
        pos_embedding = positional_enc(tokens.shape[1], tokens.shape[2])
        z_n = self.encoder_layer[0](pos_embedding + tokens)
        for encoder in self.encoder_layer:
            z_n = encoder(z_n)
        return z_n

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
    def __init__(self, dim_w_matrices, mh_size, tokens_size):
        super().__init__()
        self.mh_att = MultiHeadAttention(mh_size, tokens_size, dim_w_matrices)
        self.add_norm_l1 = AddNormalizeLayer(tokens_size)
        self.feed_forward_network = nn.Linear(tokens_size, tokens_size)
        self.add_norm_l2 = AddNormalizeLayer(tokens_size)

    def forward(self, x_n):
        z_n = self.mh_att(x_n)
        l1_out =  self.add_norm_l1(x_n, z_n)
        ffn_out = self.feed_forward_network(l1_out)
        return self.add_norm_l2(l1_out, ffn_out)

# -

# ### Self Attention
# ![attention](https://tinyurl.com/y47nyfeg)

# + pycharm={"name": "#%%\n"}


def init_weights(x_n, y_n):
    return nn.init.xavier_uniform_(torch.empty(x_n, y_n))


class MultiHeadAttention(nn.Module):
    def __init__(self, multi_head_size, tokens_size, dim_w_matrices):
        super().__init__()
        self.tokens_size = tokens_size
        self.dim_w_matrices = dim_w_matrices
        self.w_o = Parameter(init_weights(tokens_size, dim_w_matrices * multi_head_size))
        self.att_heads = nn.ModuleList()
        for _ in range(multi_head_size):
            self.att_heads.append(Attention(tokens_size,dim_w_matrices))

    def forward(self, tokens):
        z_n = []
        batch_size = tokens.shape[0]
        for head in self.att_heads:
            z_n.append(head(tokens).view(self.dim_w_matrices,-1))
        return self.w_o.mm(torch.cat(z_n)).view(batch_size, -1, self.tokens_size)


class Attention(nn.Module):
    def __init__(self, tokens_size, dim_w_matrices):
        super().__init__()
        self.tokens_size = tokens_size
        self.dim_w_matrices = dim_w_matrices

        self.w_query = Parameter(init_weights(self.tokens_size, self.dim_w_matrices))
        self.w_key = Parameter(init_weights(self.tokens_size, self.dim_w_matrices))
        self.w_vector = Parameter(init_weights(self.tokens_size, self.dim_w_matrices))

    def forward(self, tokens):
        batch_size = tokens.shape[0]
        no_batch_tokens = tokens.view(-1, self.tokens_size)
        query = no_batch_tokens.mm(self.w_query)
        key = no_batch_tokens.mm(self.w_query)
        value = no_batch_tokens.mm(self.w_query)
        current_z = F.softmax((query.mm(key.t())) / math.sqrt(self.dim_w_matrices)).mm(value)
        return current_z.view(batch_size,-1, self.dim_w_matrices)


# -

# ### Add & Normalize Layer

# + pycharm={"name": "#%%\n"}

class AddNormalizeLayer(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, residual_in, prev_res):
        xz_sum = residual_in + prev_res
        return self.layer_norm(xz_sum)
# -

# ### Positional Encoding

# + pycharm={"name": "#%%\n"}

def positional_enc(seq_len, model_dim):
    pos_emb_vector = torch.empty(seq_len, model_dim)
    for pos in range(seq_len):
        for i_col in range(model_dim):
            power_ind = 10000^(int((2*i_col)/model_dim))
            if i_col % 2 == 0:
                pos_emb_vector[pos, i_col] = math.sin(pos/power_ind)
            else:
                pos_emb_vector[pos, i_col] = math.cos(pos/power_ind)
    return pos_emb_vector


# -

# ## Dataset : Analyze & Vectorization

# ### import csv

train_csv = pd.read_csv('./input/tweet-sentiment-extraction/train.csv')
test_dt =  pd.read_csv('./input/tweet-sentiment-extraction/test.csv')
train_csv.head()

# ### split & create training, evaluation & test datasets

# + tags=[]
len_train_csv = len(train_csv)
len_test_df = len(test_dt)
total_size = len_train_csv + len_test_df

train_dt = train_csv.iloc[:int(len_train_csv*70/100)]
eval_dt = train_csv.iloc[int(len_train_csv*70/100):]

print(
"""size of train.csv file : {0}
size of test.csv file : {1}
total size : {2}

size of train datset : {3}
size of eval datset : {4}
size of test datset : {2}
""".format(
    len_train_csv,
    len_test_df,
    total_size,
    len(train_dt),
    len(eval_dt)))
# -

# ### Vectorizer

# + tags=[]
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, train_dataset, eval_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.current_dataset = self.train_dataset
        self.spacy_tokenizer = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

        self.st_voc = []
        self.vocabulary = {
            'tokens': [],
            'max_seq_len' : 0
        }
        self.len_vocabulary = len(self.vocabulary['tokens'])
        self.__init_sentiment_vocab()
        self.__init_vocab()


    def __init_sentiment_vocab(self):
        self.st_voc = ['UNK', *self.train_dataset['sentiment'].unique()]

    def __init_vocab(self):
        voc_tokens = ['UNK', 'SOS', 'EOS', 'MASK']
        max_seq_len = 0
        for feat in self.train_dataset['text']:
            tokens = [t.lemma_ for t in self.spacy_tokenizer(feat.strip())]
            voc_tokens = [*voc_tokens, *tokens]
            voc_tokens = list(set(voc_tokens))
            max_seq_len = len(tokens) if len(tokens) > max_seq_len else max_seq_len
        self.vocabulary['max_seq_len'] =  max_seq_len + 2
        self.vocabulary['tokens'] = voc_tokens


    def __getitem__(self, index):
        return {
            "text_v" : self.vectorize(self.current_dataset[index]["text"]),
            "sentiment_i" : self.get_sentiment_i(self.current_dataset[index]["sentiments"])
        }

    def __len__(self):
        return len(self.current_dataset)

    def switch_to_dataset(self,flag):
        if flag == 'train':
            self.current_dataset = self.train_dataset
        elif flag == 'eval':
            self.current_dataset = self.eval_dataset
        elif flag == 'test':
            self.current_dataset = self.test_dataset
        else:
            raise ValueError("this dataset doesn't exist !")

    def vectorize(self, tokens):
        voc = self.vocabulary['tokens']
        vector = [voc.index(t.lemma_) for t in self.spacy_tokenizer(tokens.strip())]
        vector.insert(0, voc.index('SOS'))
        vector.append(voc.index('EOS'))
        while len(vector) < self.vocabulary['max_seq_len'] :
            vector.append(voc.index('MASK'))
        return vector

    def get_sentiment_i(self, st_token):
        return  self.st_voc.index(st_token) if st_token in self.st_voc else self.st_voc.index('UNK')
# -

# ### Dataset Instanciation
twitter_dataset = TwitterDataset(train_dt, eval_dt, test_dt)
# -

# ## Parameters

# + pycharm={"name": "#%%\n"}
parameters = {
        "stack_size": 8,
        "embedding_dim": 128,
        "vocabulary_size": twitter_dataset.len_vocabulary,
        "bert_weight_matrices": 256,
        "multi_head_size": 8,
        "learning_rate": 0.0001,
        "batch_size": 128,
        "epochs": 100,
        "device": "cpu"
    }
# -

# ## Model Instanciation and DataLoader
bert = Bert(
    stack_size=parameters["stack_size"],
    embedding_dim=parameters["embedding_dim"],
    num_embeddings=parameters["vocabulary_size"],
    dim_w_matrices=parameters["bert_weight_matrices"],
    mh_size=parameters["multi_head_size"]
    )
# -


# ## Pre-Training & Fine-Tuning

# ### Task #1 : Masked LM


# ### Task #2 : Next sentence Prediction

# ### Fine Tuning

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Experimentation
