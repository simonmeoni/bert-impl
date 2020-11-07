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

# +
# ### Requirements
# **Note**: Don't forget to set the environment variable `CORPUS_SIZE`
# to set the size of corpus if it needed
# + pycharm={"name": "#%%\n"}

import os
import random
import math
import spacy
import numpy
import pandas as pd

import neptune
import torch
import torch.optim as optim
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as f
from torch.utils.data import Dataset, DataLoader

NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")

CLS = 'CLS'
MASK = 'MASK'
SEP = 'SEP'
UNK = 'UNK'
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
        embeddings = self.emb(tokens)
        pos_embedding = positional_enc(embeddings.shape[1], embeddings.shape[2])
        z_n = self.encoder_layer[0](pos_embedding + embeddings)
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
        l1_out = self.add_norm_l1(x_n, z_n)
        ffn_out = self.feed_forward_network(l1_out)
        return self.add_norm_l2(l1_out, ffn_out)


# + [markdown] pycharm={"name": "#%% md\n"}
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
            self.att_heads.append(Attention(tokens_size, dim_w_matrices))

    def forward(self, tokens):
        z_n = []
        batch_size = tokens.shape[0]
        for head in self.att_heads:
            z_n.append(head(tokens).view(self.dim_w_matrices, -1))
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
        current_z = f.softmax((query.mm(key.t())) / math.sqrt(self.dim_w_matrices), dim=1).mm(value)
        return current_z.view(batch_size, -1, self.dim_w_matrices)


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

def positional_enc(seq_len, model_dim):
    pos_emb_vector = torch.empty(seq_len, model_dim)
    for pos in range(seq_len):
        for i_col in range(model_dim):
            power_ind = 10000 ^ (int((2 * i_col) / model_dim))
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
TRAIN_PATH = './input/tweet-sentiment-extraction/train.csv'
TEST_PATH = './input/tweet-sentiment-extraction/test.csv'
if "CORPUS_SIZE" not in os.environ:
    train_csv = pd.read_csv(TRAIN_PATH)
    test_dt = pd.read_csv(TEST_PATH)
else:
    corpus_size = int(os.environ.get("CORPUS_SIZE"))
    train_csv = pd.read_csv(TRAIN_PATH)[:corpus_size]
    test_dt = pd.read_csv(TEST_PATH)[:corpus_size]
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
size of test dataset : {2}
""".format(
        len_train_csv,
        len_test_df,
        total_size,
        len(train_dt),
        len(eval_dt)))


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
        voc_tokens = [UNK, CLS, SEP, MASK]
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

    def vectorize(self, tokens):
        vector = [self.get_vocabulary_index(t.lemma_) for t in self.spacy_tokenizer(tokens.strip())]
        vector.insert(0, self.get_vocabulary_index(CLS))
        vector.append(self.get_vocabulary_index(SEP))
        while len(vector) < self.vocabulary['max_seq_len']:
            vector.append(self.get_vocabulary_index(MASK))
        return torch.LongTensor(vector)

    def get_vocabulary_index(self, token):
        tokens = self.vocabulary['tokens']
        return self.vocabulary['tokens'].index(token) if token in tokens else tokens.index(UNK)

    def get_tokens(self, tokens):
        return [self.vocabulary['tokens'][token] for token in tokens]

    def get_sentiment_i(self, st_token):
        return self.st_voc.index(st_token) if st_token in self.st_voc else self.st_voc.index(UNK)


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Dataset Instanciation
#

# + pycharm={"name": "#%%\n"}
twitter_dataset = TwitterDataset(train_dt, eval_dt, test_dt)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Parameters

# + pycharm={"name": "#%%\n"}
parameters = {
    "stack_size": 8,
    "embedding_dim": 32,
    "vocabulary_size": twitter_dataset.vocabulary['len_voc'],
    "bert_weight_matrices": 32,
    "multi_head_size": 8,
    "learning_rate": 0.0001,
    "batch_size": 5,
    "epochs": 10,
    "device": "cpu",
    "corpus test size": len(test_dt),
    "corpus train size": len(train_csv),
}

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Model Instanciation and DataLoader
#

# + pycharm={"name": "#%%\n"}
bert = Bert(
    stack_size=parameters["stack_size"],
    embedding_dim=parameters["embedding_dim"],
    num_embeddings=parameters["vocabulary_size"],
    dim_w_matrices=parameters["bert_weight_matrices"],
    mh_size=parameters["multi_head_size"]
)


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


ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert.parameters(), lr=parameters['learning_rate'])


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Pre-Training & Fine-Tuning
# For the Pre-Traning, we use instead the RoBERTa learning method.
# We use only one Pre-Training Task and we mask tokens dynamically.
# For more details to the dynamic masking
# see the original paper : https://arxiv.org/pdf/1907.11692.pdf
# -

# ### Masked LM method

# + pycharm={"name": "#%%\n"}
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
    return token not in [MASK, CLS, SEP]


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
# a pre-training classifier is needed to predict the masked token
# Bert model give only a bi contextual representation of the sentence

class PreTrainingClassifier(nn.Module):
    def __init__(self, zn_size, voc_size):
        super().__init__()
        self.classifier = nn.Linear(zn_size, voc_size)

    def forward(self, z_n):
        out = self.classifier(z_n)
        return f.softmax(out, dim=2)


# ## Pre-Training Step
# ### Training and Evaluation Loop

classifier = PreTrainingClassifier(parameters['embedding_dim'], parameters['vocabulary_size'])

# + pycharm={"name": "#%%\n"}
neptune.init('smeoni/bert-impl', api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='bert-impl-experiment', params=parameters)
for epoch in range(parameters['epochs']):
    # train loop
    twitter_dataset.switch_to_dataset("train")
    for batch in generate_batches(twitter_dataset, parameters['batch_size'],
                                  device=parameters['device']):
        x_obs = generate_batched_masked_lm(batch['vectorized_tokens'], twitter_dataset)
        y_target = batch['vectorized_tokens']
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
        neptune.send_text('train text expected', ' '.join(
            twitter_dataset.get_tokens(torch.argmax(y_pred, dim=2)[-1]))
        )
        neptune.send_text('train text observed', ' '.join(twitter_dataset.get_tokens(y_target[-1])))

    twitter_dataset.switch_to_dataset("eval")
    # evaluation loop
    for batch in generate_batches(twitter_dataset, parameters['batch_size'],
                                  device=parameters['device']):
        x_obs = generate_batched_masked_lm(batch['vectorized_tokens'], twitter_dataset)
        y_target = batch['vectorized_tokens']
        # Step 1: Compute the forward pass of the model
        bert_zn = bert(x_obs)
        y_pred = classifier(bert_zn)
        # Step 2: Compute the loss value that we wish to optimize
        loss = ce_loss(y_pred.reshape(-1, y_pred.shape[2]), y_target.reshape(-1))
        neptune.log_metric('eval loss', loss.item())

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Test Loop
# -

twitter_dataset.switch_to_dataset("test")
for batch in generate_batches(twitter_dataset, parameters['batch_size'],
                              device=parameters['device']):
    x_obs = generate_batched_masked_lm(batch['vectorized_tokens'], twitter_dataset)
    y_target = batch['vectorized_tokens']
    # Step 1: Compute the forward pass of the model
    bert_zn = bert(x_obs)
    y_pred = classifier(bert_zn)
    # Step 2: Compute the loss value that we wish to optimize
    loss = ce_loss(y_pred.reshape(-1, y_pred.shape[2]), y_target.reshape(-1))
    neptune.log_metric('test loss', loss.item())

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Experimentation
