# ---
# jupyter:
#   jupytext:
#     formats: //notebook//ipynb,//src/main//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
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

import math
import os
import random
import concurrent.futures
import re
from pathlib import Path

import neptune
import numpy
import pandas as pd
import seaborn as sns
import sentencepiece as spm
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as f
from torch.utils.data import Dataset, DataLoader
import spacy

# !python -m spacy download en_core_web_sm

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
        mask = (tokens > 0).unsqueeze(1).repeat(1, tokens.size(1), 1).unsqueeze(1)
        embeddings = self.emb(tokens)
        pos_embedding = positional_enc(embeddings.shape[1], embeddings.shape[2],
                                       self.emb.weight.device.type)
        z_n = pos_embedding + embeddings * math.sqrt(self.dim_model)
        for encoder in self.encoder_layer:
            z_n = encoder(z_n, mask)
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
        self.linear_1 = nn.Linear(dim_model, dim_model * 4)
        self.linear_2 = nn.Linear(dim_model * 4, dim_model)

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

    def forward(self, x_n, mask):
        z_n = self.mh_att(x_n, mask)
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

    def forward(self, tokens, mask):
        batch_size = tokens.shape[0]
        z_n = self.compute_attention(tokens, batch_size, mask)
        return self.linear_o(z_n.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model))

    def compute_attention(self, tokens, batch_size, mask):
        d_k = self.dim_model // self.multi_head_size
        query_mat = self.query(tokens).view(batch_size, -1, self.multi_head_size, d_k) \
            .transpose(2, 1)
        key_mat = self.key(tokens).view(batch_size, -1, self.multi_head_size, d_k) \
            .transpose(2, 1)
        value_mat = self.value(tokens).view(batch_size, -1, self.multi_head_size, d_k) \
            .transpose(2, 1)
        scores = (query_mat.matmul(key_mat.transpose(-2, -1)) / math.sqrt(self.dim_model)) \
            .masked_fill(mask == 0, 1e-11)

        return f.softmax(scores, dim=-1).matmul(value_mat)


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Add & Normalize Layer

# + pycharm={"name": "#%%\n"}

class AddNormalizeLayer(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, residual_in, prev_res):
        return residual_in + self.layer_norm(prev_res)


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


# -

# ## Import CSV files

# + pycharm={"name": "#%%\n"}
TRAIN_PATH = '../input/tweet-sentiment-extraction/train.csv'
TEST_PATH = '../input/tweet-sentiment-extraction/test.csv'
PR_TRAIN_PATH = './processed_train.csv'
PR_TEST_PATH = './processed_test.csv'
if not Path(PR_TRAIN_PATH).is_file():
    train_csv = pd.read_csv(TRAIN_PATH, dtype={'text': 'string'})
    test_dt = pd.read_csv(TEST_PATH, dtype={'text': 'string'})
else:
    train_csv = pd.read_csv(PR_TRAIN_PATH, dtype={'text': 'string'})
    test_dt = pd.read_csv(PR_TEST_PATH, dtype={'text': 'string'})
# -

# ### Cleaning and Normalization Step before Sentence Piece Training

# + pycharm={"name": "#%%\n"}
if not Path(PR_TRAIN_PATH).is_file():
    train_csv = train_csv.dropna()
    train_csv = train_csv.reset_index(drop=True)
    test_dt = test_dt.dropna()
    test_dt = test_dt.reset_index(drop=True)
    train_csv.head()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## get a word tokenisation and lemmatization for each entry

# + pycharm={"name": "#%%\n"}
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])


def processing_text(entry, dataframe, df_idx):
    text = entry['text'].lower().replace("`", "'").strip()
    text = ' '.join([token.text
                     if token.lemma_ == "-PRON-" or '*' in token.text else token.lemma_
    if not token.is_punct else '' for token in nlp(text)]).strip()
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    dataframe.at[df_idx, 'text'] = re.sub(r'\s\s+', ' ', text)


def processing_df(dataframe, path):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(processing_text, df_entry, dataframe, df_idx):
                         df_entry for df_idx, df_entry in enumerate(dataframe.iloc)}
    for _ in concurrent.futures.as_completed(future_to_url):
        pass
    dataframe.to_csv(path)


if not Path(PR_TRAIN_PATH).is_file():
    processing_df(test_dt, PR_TEST_PATH)
    processing_df(train_csv, PR_TRAIN_PATH)
train_csv.head()
# -

# ## Train & Initialize Sentence Piece

# + pycharm={"name": "#%%\n"}
PATH = './tweet-sentiment-extraction'
with open(PATH + '.txt', 'w') as voc_txt:
    for t in train_csv['text']:
        voc_txt.write(t + '\n')
SPM_ARGS = "" \
           "--input={0}.txt " \
           "--model_prefix={0} " \
           "--pad_id=0 " \
           "--unk_id=1 " \
           "--bos_id=2 " \
           "--eos_id=3 " \
           "--pad_piece={1} " \
           "--unk_piece={2} " \
           "--bos_piece={3} " \
           "--eos_piece={4}" \
    .format(PATH, PAD, UNK, CLS, SEP)
spm.SentencePieceTrainer.Train(SPM_ARGS)
sp = spm.SentencePieceProcessor()
sp.Load(PATH + '.model')
print(sp.EncodeAsPieces('this is a test'))
print(sp.EncodeAsIds('this is a test'))
# -

# ## Dataset : Analyze & Vectorization

# ### resize the corpus if it needed

# + pycharm={"name": "#%%\n"}
if "CORPUS_SIZE" in os.environ:
    corpus_size = int(os.environ.get("CORPUS_SIZE"))
    train_csv = train_csv[:corpus_size]
    test_dt = test_dt[:corpus_size]
else:
    train_csv = train_csv[:100]
    test_dt = test_dt[:10]

# -

# ### analysis

# + pycharm={"name": "#%%\n"}
train_csv['sequence length'] = ''
URL_COUNT = 0
for idx, d in enumerate(train_csv.iloc):
    train_csv.at[idx, 'sequence length'] = len(sp.EncodeAsIds(d['text']))
for idx, d in enumerate(test_dt.iloc):
    test_dt.at[idx, 'sequence length'] = len(sp.EncodeAsIds(d['text']))
sns.set(font_scale=2)
sns.displot(x='sequence length', data=train_csv, aspect=2, height=20)
print('number of entries containing a url : ' + str(URL_COUNT))
print('number of entries in train.csv : ' + str(len(train_csv)))
# -

# ### Filter the entries containing url and the less frequent length sequences

# + pycharm={"name": "#%%\n"}
del train_csv['selected_text']
train_csv = train_csv.drop(train_csv[train_csv['sequence length'].ge(35)].index)
train_csv = train_csv.drop(train_csv[train_csv['sequence length'].le(5)].index)
train_csv = train_csv.reset_index(drop=True)
print('number of entries in train.csv after filtering : ' + str(len(train_csv)))
sns.displot(x='sequence length', data=train_csv, aspect=2, height=20)

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
    def __init__(self, train_dataset, eval_dataset, test_dataset, sentence_piece):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.current_dataset = self.train_dataset
        self.test_dataset = test_dataset
        self.sentence_piece = sentence_piece
        self.st_voc = []
        self.max_seq_len = int(pd.concat(
            [train_dataset, eval_dataset, test_dataset])['sequence length'].max()) + 2
        self.__init_sentiment_vocab()

    def __init_sentiment_vocab(self):
        self.st_voc = [UNK, *self.train_dataset['sentiment'].unique()]

    def get_vocab_size(self):
        return self.sentence_piece.vocab_size() + 1

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
        vector = self.sentence_piece.EncodeAsIds(tokens)
        return torch.LongTensor(
            [sp.bos_id()] + vector + [sp.eos_id()] +
            [self.get_pad()] * (self.max_seq_len - len(vector) - 2)
        )

    def get_mask(self):
        return self.sentence_piece.vocab_size()

    def get_pad(self):
        return self.sentence_piece.pad_id()

    def get_cls(self):
        return self.sentence_piece.bos_id()

    def get_sep(self):
        return self.sentence_piece.eos_id()

    def get_tokens(self, ids):
        return ' '.join([self.sentence_piece.Decode(i) if i != self.get_mask()
                         else MASK for i in ids.tolist()]).strip()

    def get_sentiment_i(self, st_token):
        return self.st_voc.index(st_token) if st_token in self.st_voc else self.st_voc.index(UNK)


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Dataset Instantiation
#

# + pycharm={"name": "#%%\n"}
twitter_dataset = TwitterDataset(train_dt, eval_dt, test_dt, sp)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Parameters

# + pycharm={"name": "#%%\n"}
parameters = {
    "stack_size": 6,
    "vocabulary_size": twitter_dataset.get_vocab_size(),
    "bert_dim_model": 256,
    "multi_heads": 8,
    "pre_train_learning_rate": 1e-4,
    "st_learning_rate": 2e-5,
    "batch_size": 1,
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
    padding_idx=twitter_dataset.get_pad()
).to(current_device)

ce_loss = nn.CrossEntropyLoss(ignore_index=twitter_dataset.get_pad()) \
    .to(current_device)


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
        if numpy.random.uniform() < mask_prob and is_not_markers(idx_token, dataset)
        else idx_token
        for idx_token in vector
    ])


# -

def is_not_markers(token, dataset):
    return token not in [dataset.get_cls(), dataset.get_sep(),
                         dataset.get_pad(), dataset.get_mask()]


def replace_token(token, dataset, rnd_t_prob, unchanged_prob):
    prob = numpy.random.uniform()
    if prob < rnd_t_prob:
        return replace_by_another_id(token, dataset)
    if rnd_t_prob < prob < unchanged_prob + rnd_t_prob:
        return token
    return dataset.get_mask()


def replace_by_another_id(index_token, dataset):
    replaced_index_t = index_token
    not_include_t = [
        dataset.get_cls(),
        dataset.get_sep(),
        dataset.get_mask(),
        dataset.get_pad(),
        index_token
    ]
    while replaced_index_t in not_include_t:
        replaced_index_t = random.choice(range(twitter_dataset.get_vocab_size()))
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

    def forward(self, z_n):
        return self.l_1(z_n)


# ## Pre-Training Step
# ### Training and Evaluation Loop

optimizer = optim.Adam(bert.parameters(), lr=parameters['pre_train_learning_rate'])

pre_train_classifier = PreTrainingClassifier(parameters['bert_dim_model'],
                                             parameters['vocabulary_size']).to(current_device)

# + pycharm={"name": "#%%\n"}
if "TEST_ENV" not in os.environ.keys():
    neptune.init('smeoni/bert-impl', api_token=NEPTUNE_API_TOKEN)
    neptune.create_experiment(name='bert-impl-experiment', params=parameters)
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
            y_pred = pre_train_classifier(bert_zn)
            # Step 3: Compute the loss value that we wish to optimize
            loss = ce_loss(y_pred.reshape(-1, y_pred.shape[2]), y_target.reshape(-1))
            # Step 4: Propagate the loss signal backward
            loss.backward()
            # Step 5: Trigger the optimizer to perform one update
            optimizer.step()
            neptune.log_metric('pre-train loss', loss.item())
            observed_ids = torch.argmax(y_pred, dim=2)[-1]
            RAW_TEXT_OBSERVED = sp.Decode([id_obv for id_obv in observed_ids.tolist()
                                           if id_obv != twitter_dataset.get_mask()])
            neptune.send_text('raw pre-train text observed', RAW_TEXT_OBSERVED)
            RAW_TEXT_EXPECTED = sp.Decode(y_target[-1].tolist())
            neptune.send_text('raw pre-train text expected', RAW_TEXT_EXPECTED)


# -

# ## Fine-Tuning Step
# ### Fine-Tuning Classifier

# + pycharm={"name": "#%%\n"}
class FineTuningClassifier(nn.Module):
    def __init__(self, zn_size, st_voc_size, voc_size):
        super().__init__()
        self.l_1 = nn.Linear(zn_size, voc_size)
        self.l_2 = nn.Linear(voc_size, st_voc_size)

    def forward(self, z_n):
        l1_out = f.relu((self.l_1(z_n)))
        out = self.l_2(l1_out)
        return out


# -

# ### Fine-Tuning Training Loop

# + pycharm={"name": "#%%\n"}
fine_tuning_classifier = FineTuningClassifier(parameters['bert_dim_model'],
                                              len(twitter_dataset.st_voc),
                                              parameters['vocabulary_size']).to(current_device)

optimizer = optim.Adam(bert.parameters(), lr=parameters['st_learning_rate'])


def no_learn_loop(corpus, model, no_learn_loss, dataset, no_learn_device):
    dataset.switch_to_dataset(corpus)
    # evaluation loop
    for no_learn_batch in generate_batches(dataset, parameters['batch_size'],
                                           device=no_learn_device):
        no_learn_x_obs = generate_batched_masked_lm(no_learn_batch['vectorized_tokens'], dataset) \
            .to(no_learn_device)
        no_learn_y_target = no_learn_batch['sentiment_i'].to(no_learn_device)
        # Step 1: Compute the forward pass of the model
        no_learn_zn = model(no_learn_x_obs)
        no_learn_y_pred = fine_tuning_classifier(no_learn_zn[:, -1, :])
        # Step 2: Compute the loss value that we wish to optimize
        no_ll_res = no_learn_loss(no_learn_y_pred, no_learn_y_target.reshape(-1))

        neptune.log_metric('sentiment ' + corpus + ' loss', no_ll_res.item())
        neptune.send_text('sentiment ' + corpus + ' text',
                          sp.Decode(x_obs[-1].tolist()))
        neptune.send_text('sentiment' + corpus + ' observed',
                          twitter_dataset.st_voc[torch.argmax(y_pred, dim=-1)[-1]])
        neptune.send_text('sentiment ' + corpus + ' expected', twitter_dataset.st_voc[y_target[-1]])


if "TEST_ENV" not in os.environ.keys():
    for epoch in range(parameters['epochs']):
        # train loop
        twitter_dataset.switch_to_dataset("train")
        for batch in generate_batches(twitter_dataset,
                                      parameters['batch_size'],
                                      device=parameters['device']):
            x_obs = batch['vectorized_tokens'].to(current_device)
            y_target = batch['sentiment_i'].to(current_device)
            # Step 1: Clear the gradients
            bert.zero_grad()
            # Step 2: Compute the forward pass of the model
            bert_zn = bert(x_obs)
            y_pred = fine_tuning_classifier(bert_zn[:, -1, :])
            # Step 3: Compute the loss value that we wish to optimize
            loss = ce_loss(y_pred, y_target.reshape(-1))
            # Step 4: Propagate the loss signal backward
            loss.backward()
            # Step 5: Trigger the optimizer to perform one update
            optimizer.step()
            neptune.log_metric('sentiment train loss', loss.item())
            neptune.send_text('sentiment train text', sp.Decode(x_obs[-1].tolist()))
            neptune.send_text('sentiment train observed',
                              twitter_dataset.st_voc[torch.argmax(y_pred, dim=-1)[-1]])
            neptune.send_text('sentiment train expected',
                              twitter_dataset.st_voc[y_target[-1]])

        no_learn_loop('eval', bert, ce_loss, twitter_dataset, parameters['device'])

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Fine-Tuning Test Loop

# + pycharm={"name": "#%%\n"}
if "TEST_ENV" not in os.environ.keys():
    no_learn_loop('test', bert, ce_loss, twitter_dataset, parameters['device'])
    neptune.stop()
