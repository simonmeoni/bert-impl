import os
from pathlib import Path

import neptune
import pandas as pd
import sentencepiece as spm
import spacy
import torch
from torch import nn, optim

from src.bert_impl.dataset.bert_twitter_dataset import TwitterDataset
from src.bert_impl.model.bert.bert import Bert
from src.bert_impl.train.loop import pre_train_loop, fine_tuning_loop
from src.bert_impl.utils.utils import NEPTUNE_API_TOKEN, PR_TRAIN_PATH, TRAIN_PATH, TEST_PATH, \
    PR_TEST_PATH, processing_df, PAD, UNK, \
    CLS, SEP, set_seq_length

# set to cuda device if it is available
if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
else:
    TORCH_DEVICE = "cpu"

current_device = torch.device(TORCH_DEVICE)

# initialize corpus
if not Path(PR_TRAIN_PATH).is_file():
    train_csv = pd.read_csv(TRAIN_PATH, dtype={'text': 'string'})
    test_dt = pd.read_csv(TEST_PATH, dtype={'text': 'string'})
else:
    train_csv = pd.read_csv(PR_TRAIN_PATH, dtype={'text': 'string'})
    test_dt = pd.read_csv(PR_TEST_PATH, dtype={'text': 'string'})

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

if not Path(PR_TRAIN_PATH).is_file():
    test_dt = processing_df(test_dt, PR_TEST_PATH, nlp)
    train_csv = processing_df(train_csv, PR_TRAIN_PATH, nlp)

PATH = './src/resources/tweet-sentiment-extraction'
with open(PATH + '.txt', 'w') as voc_txt:
    for t in train_csv['text']:
        voc_txt.write(t + '\n')

# initialize sentence piece
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
           "--token_size=8000" \
    .format(PATH, PAD, UNK, CLS, SEP)
spm.SentencePieceTrainer.Train(SPM_ARGS)
sp = spm.SentencePieceProcessor()
sp.Load(PATH + '.model')


# Set the corpus size
if "CORPUS_SIZE" in os.environ:
    corpus_size = int(os.environ.get("CORPUS_SIZE"))
    train_csv = train_csv[:corpus_size]
    test_dt = test_dt[:corpus_size]
else:
    train_csv = train_csv[:100]
    test_dt = test_dt[:10]

# set the length of the different entries and remove certain cases
set_seq_length(train_csv, sp)

train_csv = train_csv.drop(train_csv[train_csv['sequence length'].ge(35)].index)
train_csv = train_csv.drop(train_csv[train_csv['sequence length'].le(5)].index)
train_csv = train_csv.reset_index(drop=True)

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

# set dataset
twitter_dt = TwitterDataset(train_dt, eval_dt, test_dt, sp)

# set parameters
parameters = {
    "stack_size": 6,
    "vocabulary_size": twitter_dt.get_vocab_size(),
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

# set the model
bert = Bert(
    stack_size=parameters["stack_size"],
    voc_size=parameters["vocabulary_size"],
    dim_model=parameters["bert_dim_model"],
    mh_size=parameters["multi_heads"],
    padding_idx=twitter_dt.get_pad()
).to(current_device)


loss = nn.CrossEntropyLoss(ignore_index=twitter_dt.get_pad()) \
    .to(current_device)

optimizer = optim.Adam(bert.parameters(), lr=parameters['pre_train_learning_rate'])
parameters['model'] = bert
parameters['optimizer'] = optimizer
parameters['loss'] = loss
neptune.init('smeoni/bert-impl', api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='bert_impl-experiment', params=parameters)

# Pre-Training
pre_train_loop(neptune, twitter_dt, True, **parameters)

# Fine-Tuning
optimizer = optim.Adam(bert.parameters(), lr=parameters['st_learning_rate'])
parameters['optimizer'] = optimizer
fine_tuning_loop(neptune, twitter_dt, True, **parameters)
