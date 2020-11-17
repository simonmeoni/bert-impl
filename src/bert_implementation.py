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

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Requirements
# **Note**: Don't forget to set the environment variable `CORPUS_SIZE`
# to set the size of corpus if it needed
# + pycharm={"name": "#%%\n"}

import os
import random
import math
import re

import spacy
import numpy
import pandas as pd

import neptune
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as f
from torch.utils.data import Dataset, DataLoader

NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")

CLS = 'CLS'
MASK = 'MASK'
SEP = 'SEP'
PAD = 'PAD'
UNK = 'UNK'

# enable cuda if it exists
if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
else:
    TORCH_DEVICE = "cpu"

current_device = torch.device(TORCH_DEVICE)
# -

# ### Bert Encoder Stacks
# * Bert takes as input a sequence of plain text tokens
# * the output is a representation vector of the size of the hidden layers
# * Bert is a stack of multi-layer bidirectional Transformer encoder

# + pycharm={"name": "#%%\n"}


class Bert(nn.Module):
    # pylint: disable=too-many-arguments
    def __init__(self, stack_size, voc_size,
                 dim_model, mh_size, padding_idx=0):
        super().__init__()
        self.dim_model = dim_model
        self.emb = nn.Embedding(
            embedding_dim=dim_model,
            num_embeddings=voc_size,
            padding_idx=padding_idx
        )
        self.encoder_layer = nn.ModuleList()
        for _ in range(stack_size):
            self.encoder_layer.append(Encoder(dim_model, mh_size))

    def forward(self, tokens):
        embeddings = self.emb(tokens)
        pos_embedding = positional_enc(embeddings.shape[1], embeddings.shape[2],
                                       self.emb.weight.device.type)
        z_n = self.encoder_layer[0](pos_embedding + embeddings * math.sqrt(self.dim_model))
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

class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_model)
        self.linear_2 = nn.Linear(dim_model, dim_model)

    def forward(self, x_n):
        out_l1 = f.relu(self.linear_1(x_n))
        return self.linear_2(out_l1)


class Encoder(nn.Module):
    def __init__(self, dim_model, mh_size):
        super().__init__()
        self.mh_att = MultiHeadAttention(mh_size, dim_model)
        self.add_norm_l1 = AddNormalizeLayer(dim_model)
        self.feed_forward_network = FeedForwardNetwork(dim_model)
        self.add_norm_l2 = AddNormalizeLayer(dim_model)

    def forward(self, x_n):
        z_n = self.mh_att(x_n)
        l1_out = self.add_norm_l1(x_n, z_n)
        ffn_out = self.feed_forward_network(l1_out)
        return self.add_norm_l2(l1_out, ffn_out)


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Self Attention
# ![attention](https://tinyurl.com/y47nyfeg)

# + pycharm={"name": "#%%\n"}
class MultiHeadAttention(nn.Module):
    def __init__(self, multi_head_size, dim_model):
        super().__init__()
        self.dim_model = dim_model
        self.multi_head_size = multi_head_size
        self.linear_o = nn.Linear(self.dim_model, self.dim_model)
        self.query = nn.Linear(self.dim_model, self.dim_model)
        self.key = nn.Linear(self.dim_model, self.dim_model)
        self.value = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, tokens):
        batch_size = tokens.shape[0]
        z_n = self.compute_attention(tokens, batch_size)
        return self.linear_o(z_n.view(batch_size, -1, self.dim_model))

    def compute_attention(self, tokens, batch_size):
        d_k = self.dim_model // self.multi_head_size
        query_mat = self.query(tokens).view(batch_size, -1, self.multi_head_size, d_k)
        key_mat = self.key(tokens).view(batch_size, -1, self.multi_head_size, d_k)
        value_mat = self.value(tokens).view(batch_size, -1, self.multi_head_size, d_k)
        return f.softmax(
            (query_mat.matmul(key_mat.transpose(-2, -1)) / math.sqrt(self.dim_model)), dim=-1
        ).matmul(value_mat)


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Add & Normalize Layer

# + pycharm={"name": "#%%\n"}

class AddNormalizeLayer(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, residual_in, prev_res):
        xz_sum = residual_in + prev_res
        return self.layer_norm(xz_sum)


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Positional Encoding

# + pycharm={"name": "#%%\n"}

def positional_enc(seq_len, model_dim, device="cpu"):
    pos_emb_vector = torch.empty(seq_len, model_dim).to(device)
    for pos in range(seq_len):
        for i_col in range(model_dim):
            power_ind = 10000 ** ((2 * i_col) / model_dim)
            if i_col % 2 == 0:
                pos_emb_vector[pos, i_col] = math.sin(pos / power_ind)
            else:
                pos_emb_vector[pos, i_col] = math.cos(pos / power_ind)
    return pos_emb_vector


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Dataset : Analyze & Vectorization
# -

# ### import csv

# + pycharm={"name": "#%%\n"}
TRAIN_PATH = '../input/tweet-sentiment-extraction/train.csv'
TEST_PATH = '../input/tweet-sentiment-extraction/test.csv'
if "CORPUS_SIZE" not in os.environ:
    train_csv = pd.read_csv(TRAIN_PATH, dtype={'text': 'string'})[:100]
    test_dt = pd.read_csv(TEST_PATH, dtype={'text': 'string'})[:10]
else:
    corpus_size = int(os.environ.get("CORPUS_SIZE"))
    train_csv = pd.read_csv(TRAIN_PATH, dtype={'text': 'string'})[:corpus_size]
    test_dt = pd.read_csv(TEST_PATH, dtype={'text': 'string'})[:corpus_size]
train_csv = train_csv[pd.notnull(train_csv['text'])]
test_dt = test_dt[pd.notnull(test_dt['text'])]
train_csv.head()

# + [markdown] pycharm={"name": "#%% md\n"}
# ### split & create training, evaluation & test datasets

# + pycharm={"name": "#%%\n"} tags=[]
len_train_csv = len(train_csv)
len_test_df = len(test_dt)
total_size = len_train_csv + len_test_df

train_dt = train_csv.iloc[:int(len_train_csv * 70 / 100)]
eval_dt = train_csv.iloc[int(len_train_csv * 70 / 100):]

print(
    """size of train.csv file : {0}
size of test.csv file : {1}
total size : {2}

size of train dataset : {3}
size of eval dataset : {4}
size of test dataset : {5}
""".format(
        len_train_csv,
        len_test_df,
        total_size,
        len(train_dt),
        len(eval_dt),
        len(test_dt)))


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Vectorizer

# + pycharm={"name": "#%%\n"} tags=[]
class TwitterDataset(Dataset):
    def __init__(self, train_dataset, eval_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.current_dataset = self.train_dataset
        self.spacy_tokenizer = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

        self.st_voc = []
        self.vocabulary = {
            'tokens': [],
            'max_seq_len': 0,
            'len_voc': 0
        }
        self.__init_sentiment_vocab()
        self.__init_vocab()

    def __init_sentiment_vocab(self):
        self.st_voc = [UNK, *self.train_dataset['sentiment'].unique()]

    def __init_vocab(self):
        voc_tokens = [UNK, CLS, SEP, MASK, PAD]
        max_seq_len = 0
        for feat in self.train_dataset['text']:
            if not isinstance(feat, float):
                tokens, max_seq_len = self.extract_tokens(feat, max_seq_len)
                voc_tokens = [*voc_tokens, *tokens]
                voc_tokens = list(set(voc_tokens))
        for feat in pd.concat([self.eval_dataset['text'], self.test_dataset['text']]):
            if not isinstance(feat, float):
                _, max_seq_len = self.extract_tokens(feat, max_seq_len)

        self.vocabulary['max_seq_len'] = max_seq_len + 2
        self.vocabulary['tokens'] = voc_tokens
        self.vocabulary['len_voc'] = len(self.vocabulary['tokens'])

    def extract_tokens(self, feat, max_seq_len):
        tokens = [t.lemma_ for t in self.spacy_tokenizer(feat.strip())]
        max_seq_len = len(tokens) if len(tokens) > max_seq_len else max_seq_len
        return tokens, max_seq_len

    def __getitem__(self, index):
        return {
            'vectorized_tokens': self.vectorize(self.current_dataset.iloc[index]["text"]),
            "sentiment_i": self.get_sentiment_i(self.current_dataset.iloc[index]["sentiment"])
        }

    def __len__(self):
        return len(self.current_dataset)

    def switch_to_dataset(self, flag):
        if flag == 'train':
            self.current_dataset = self.train_dataset
        elif flag == 'eval':
            self.current_dataset = self.eval_dataset
        elif flag == 'test':
            self.current_dataset = self.test_dataset
        else:
            raise ValueError("this dataset doesn't exist !")

    # noinspection PyArgumentList
    def vectorize(self, tokens):
        vector = [self.get_vocabulary_index(t.lemma_) for t in self.spacy_tokenizer(tokens.strip())]
        vector.insert(0, self.get_vocabulary_index(CLS))
        vector.append(self.get_vocabulary_index(SEP))
        while len(vector) < self.vocabulary['max_seq_len']:
            vector.append(self.get_vocabulary_index(PAD))
        return torch.LongTensor(vector)

    def get_vocabulary_index(self, token):
        tokens = self.vocabulary['tokens']
        return self.vocabulary['tokens'].index(token) if token in tokens else tokens.index(UNK)

    def get_tokens(self, tokens):
        return [self.vocabulary['tokens'][token] for token in tokens]

    def get_sentiment_i(self, st_token):
        return self.st_voc.index(st_token) if st_token in self.st_voc else self.st_voc.index(UNK)


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Dataset Instantiation
#

# + pycharm={"name": "#%%\n"}
twitter_dataset = TwitterDataset(train_dt, eval_dt, test_dt)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Parameters

# + pycharm={"name": "#%%\n"}
parameters = {
    "stack_size": 8,
    "vocabulary_size": twitter_dataset.vocabulary['len_voc'],
    "bert_dim_model": 256,
    "multi_heads": 8,
    "learning_rate": 0.001,
    "batch_size": 5,
    "epochs": 100,
    "device": current_device,
    "corpus test size": len(test_dt),
    "corpus train size": len(train_csv),
}

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Model Instantiation and DataLoader
#

# + pycharm={"name": "#%%\n"}
bert = Bert(
    stack_size=parameters["stack_size"],
    voc_size=parameters["vocabulary_size"],
    dim_model=parameters["bert_dim_model"],
    mh_size=parameters["multi_heads"],
    padding_idx=twitter_dataset.get_vocabulary_index('PAD')
).to(current_device)


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
    ensure each tensor is on the write device location.
    """
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=drop_last)

    for data_dict in data_loader:
        data = {}
        for name, _ in data_dict.items():
            data[name] = data_dict[name].to(device)
        yield data


ce_loss = nn.CrossEntropyLoss(ignore_index=twitter_dataset.get_vocabulary_index('PAD'))\
    .to(current_device)
optimizer = optim.Adam(bert.parameters(), lr=parameters['learning_rate'])


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Pre-Training & Fine-Tuning
# For the Pre-Training, we use instead the RoBERTa learning method.
# We use only one Pre-Training Task and we mask tokens dynamically.
# For more details to the dynamic masking
# see the original paper : https://arxiv.org/pdf/1907.11692.pdf
# -

# ### Masked LM method

# + pycharm={"name": "#%%\n"}
# noinspection PyArgumentList
def generate_masked_lm(vector, dataset, mask_prob=.15, rnd_t_prob=.1, unchanged_prob=.1):
    return torch.LongTensor([
        replace_token(idx_token, dataset, rnd_t_prob, unchanged_prob)
        if numpy.random.uniform() < mask_prob
        and is_not_markers(dataset.vocabulary['tokens'][idx_token])
        else idx_token
        for idx_token in vector
    ])


# -

def is_not_markers(token):
    return token not in [MASK, CLS, SEP, PAD]


def replace_token(token, dataset, rnd_t_prob, unchanged_prob):
    prob = numpy.random.uniform()
    if prob < rnd_t_prob:
        return replace_by_another_token(token, dataset)
    if rnd_t_prob < prob < unchanged_prob + rnd_t_prob:
        return token
    return dataset.vocabulary['tokens'].index(MASK)


def replace_by_another_token(index_token, dataset):
    replaced_index_t = index_token
    not_include_t = [
        dataset.get_vocabulary_index(CLS),
        dataset.get_vocabulary_index(SEP),
        dataset.get_vocabulary_index(MASK),
        dataset.get_vocabulary_index(PAD),
        index_token
    ]
    while replaced_index_t in not_include_t:
        replaced_token = random.choice(dataset.vocabulary['tokens'])
        replaced_index_t = dataset.vocabulary['tokens'].index(replaced_token)
    return replaced_index_t


def generate_batched_masked_lm(batched_vectors, dataset,
                               mask_prob=.15, rnd_t_prob=.1, unchanged_prob=.1):
    batched_masked_lm = [
        generate_masked_lm(vector, dataset, mask_prob, rnd_t_prob, unchanged_prob)
        for vector in batched_vectors
    ]
    return torch.stack(batched_masked_lm)


# ### Pre-Training Classifier
# a pre-training l_1 is needed to predict the masked token
# Bert model give only a bi contextual representation of the sentence

class PreTrainingClassifier(nn.Module):
    def __init__(self, zn_size, voc_size):
        super().__init__()
        self.l_1 = nn.Linear(zn_size, voc_size)
        self.l_2 = nn.Linear(voc_size, voc_size)

    def forward(self, z_n):
        l1_out = f.relu((self.l_1(z_n)))
        out = self.l_2(l1_out)
        return out


# ## Pre-Training Step
# ### Training and Evaluation Loop

classifier = PreTrainingClassifier(parameters['bert_dim_model'], parameters['vocabulary_size'])\
    .to(current_device)

# + pycharm={"name": "#%%\n"}
neptune.init('smeoni/bert-impl', api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='bert-impl-experiment', params=parameters)


def no_learning_loop(corpus, model, no_learn_loss, dataset, no_learn_device):
    dataset.switch_to_dataset(corpus)
    # evaluation loop
    for no_learn_batch in generate_batches(dataset, parameters['batch_size'],
                                  device=no_learn_device):
        no_learn_x_obs = generate_batched_masked_lm(no_learn_batch['vectorized_tokens'], dataset)\
            .to(no_learn_device)
        no_learn_y_target = no_learn_batch['vectorized_tokens'].to(no_learn_device)
        # Step 1: Compute the forward pass of the model
        no_learn_zn = model(no_learn_x_obs)
        no_learn_y_pred = classifier(no_learn_zn)
        # Step 2: Compute the loss value that we wish to optimize
        no_learn_loss = no_learn_loss(no_learn_y_pred.reshape(-1, no_learn_y_pred.shape[2]),
                                      no_learn_y_target.reshape(-1))
        neptune.log_metric(corpus + 'loss', no_learn_loss.item())


for epoch in range(parameters['epochs']):
    # train loop
    twitter_dataset.switch_to_dataset("train")
    for batch in generate_batches(twitter_dataset,
                                  parameters['batch_size'],
                                  device=parameters['device']):
        x_obs = generate_batched_masked_lm(batch['vectorized_tokens'],
                                           twitter_dataset).to(current_device)
        y_target = batch['vectorized_tokens'].to(current_device)
        # Step 1: Clear the gradients
        bert.zero_grad()
        # Step 2: Compute the forward pass of the model
        bert_zn = bert(x_obs)
        y_pred = classifier(bert_zn)
        # Step 3: Compute the loss value that we wish to optimize
        loss = ce_loss(y_pred.reshape(-1, y_pred.shape[2]), y_target.reshape(-1))
        # Step 4: Propagate the loss signal backward
        loss.backward()
        # Step 5: Trigger the optimizer to perform one update
        optimizer.step()
        neptune.log_metric('train loss', loss.item())
        RAW_TEXT_OBSERVED = ' '.join(twitter_dataset.get_tokens(torch.argmax(y_pred, dim=2)[-1]))
        neptune.send_text('raw train text observed', RAW_TEXT_OBSERVED
                          )
        RAW_TEXT_EXPECTED = ' '.join(twitter_dataset.get_tokens(y_target[-1]))
        neptune.send_text('raw train text expected', RAW_TEXT_EXPECTED)

        PATTERN = "CLS (.*?) SEP"
        neptune.send_text('clean train text observed',
                          re.search(PATTERN, RAW_TEXT_OBSERVED).group(1)
                          if re.search(PATTERN, RAW_TEXT_OBSERVED) else 'there is no markers ! ')
        neptune.send_text('clean train text expected',
                          re.search(PATTERN, RAW_TEXT_EXPECTED).group(1))

    no_learning_loop('eval', bert, ce_loss, twitter_dataset, parameters['device'])

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Test Loop
# -
no_learning_loop('test', bert, ce_loss, twitter_dataset, parameters['device'])
# + [markdown] pycharm={"name": "#%% md\n"}
# ## Experimentation
