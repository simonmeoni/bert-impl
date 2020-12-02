import os
from pathlib import Path

import neptune
import pandas as pd
import sentencepiece as spm
import spacy
import torch
from numpy import mean
from sklearn.model_selection import KFold
from torch import nn, optim

from src.bert_impl.dataset.bert_twitter_dataset import TwitterDataset
from src.bert_impl.model.bert.bert import Bert
from src.bert_impl.train.loop import pre_train_loop, fine_tuning_loop
from src.bert_impl.utils.utils import NEPTUNE_API_TOKEN, PR_TRAIN_PATH, TRAIN_PATH, processing_df, \
    PAD, UNK, \
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
else:
    train_csv = pd.read_csv(PR_TRAIN_PATH, dtype={'text': 'string'})

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

if not Path(PR_TRAIN_PATH).is_file():
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
else:
    train_csv = train_csv[:80]

# set the length of the different entries and remove certain cases
set_seq_length(train_csv, sp)

train_csv = train_csv.drop(train_csv[train_csv['sequence length'].ge(35)].index)
train_csv = train_csv.drop(train_csv[train_csv['sequence length'].le(5)].index)
train_csv = train_csv.reset_index(drop=True)

print("size of the dataset : {0}".format(len(train_csv)))

# set parameters
parameters = {
    "stack_size": 6,
    "vocabulary_size": sp.vocab_size() + 1,
    "bert_dim_model": 256,
    "multi_heads": 8,
    "learning_rate": 1e-4,
    "st_learning_rate": 2e-5,
    "batch_size": 2,
    "epochs": 100,
    "device": current_device,
    "corpus train size": len(train_csv),
    "folds": 8
}

loss = nn.CrossEntropyLoss(ignore_index=sp.pad_id()) \
    .to(current_device)

neptune.init('smeoni/bert-impl', api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='bert_impl-experiment', params=parameters)

folds = KFold(n_splits=parameters['folds'], shuffle=False)
cv_pt = []
cv_ft = []
for fold in folds.split(train_csv):
    # set the model
    bert = Bert(
        stack_size=parameters["stack_size"],
        voc_size=parameters["vocabulary_size"],
        dim_model=parameters["bert_dim_model"],
        mh_size=parameters["multi_heads"],
        padding_idx=sp.pad_id()
    ).to(current_device)
    parameters['model'] = bert
    parameters['optimizer'] = optim.Adam(bert.parameters(),
                                         lr=parameters['learning_rate'])
    parameters['st_optimizer'] = optim.Adam(bert.parameters(), lr=parameters['st_learning_rate'])
    parameters['loss'] = loss

    train_fold, eval_fold = fold
    train_dt = TwitterDataset(train_csv.iloc[train_fold], sp)
    eval_dt = TwitterDataset(train_csv.iloc[eval_fold], sp)
    pre_train_loop(neptune, train_dt, train=True, **parameters)
    fine_tuning_loop(neptune, train_dt, True, **parameters)
    cv_score_pt = pre_train_loop(neptune, eval_dt, train=False, **parameters)
    cv_score_ft = fine_tuning_loop(neptune, eval_dt, train=False, **parameters)
    neptune.log_metric('pre-training cross validation', cv_score_pt)
    neptune.log_metric('fine tuning cross validation', cv_score_ft)
    cv_pt.append(cv_score_pt)
    cv_ft.append(cv_score_ft)

print("""
cross validation score mean :
* pre-training : {0}
* fine-tuning :  {1}
cross validation scores : 
* pre-training : {2}
* fine-tuning :  {3}

""".format(mean(cv_pt), mean(cv_ft), cv_pt, cv_ft))
