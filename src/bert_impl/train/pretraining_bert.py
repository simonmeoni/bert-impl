from pathlib import Path

import neptune
import pandas as pd
import sentencepiece as spm
import spacy
import torch
from torch import nn, optim

from src.bert_impl.dataset.bert_twitter_dataset import TwitterDataset
from src.bert_impl.model.bert.bert import Bert
from src.bert_impl.train.loop import pre_train_loop
from src.bert_impl.utils.utils import processing_df, \
    set_seq_length, create_sp_model


# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def pretrain_bert_model(train_path,
                        test_path,
                        pretrain_path,
                        checkpoint_path,
                        sp_path,
                        neptune_api_token,
                        stack_size,
                        bert_dim_model,
                        head_size,
                        pt_lr,
                        batch_size,
                        epochs,
                        corpus_size):
    # set to cuda device if it is available
    if torch.cuda.is_available():
        torch_device = "cuda"
    else:
        torch_device = "cpu"
    current_device = torch.device(torch_device)
    # initialize corpus
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
    if not Path(pretrain_path).is_file():
        train_csv = pd.read_csv(train_path, dtype={'text': 'string'})
        del train_csv['selected_text']
        test_csv = pd.read_csv(test_path, dtype={'text': 'string'})
        pretrain_dt = processing_df(pd.concat([train_csv, test_csv], ignore_index=True), nlp)
        pretrain_dt.to_csv(pretrain_path, index=False)
    else:
        pretrain_dt = pd.read_csv(pretrain_path, dtype={'text': 'string'})

    sentence_piece = create_sp_model(pretrain_dt, sp_path, spm)
    sentence_piece.Load(sp_path + '.model')

    # Set the corpus size
    if int(corpus_size) >= 0:
        pretrain_dt = pretrain_dt[:int(corpus_size)]
    # set the length of the different entries and remove certain cases
    set_seq_length(pretrain_dt, sentence_piece)
    print("size of the dataset : {0}".format(len(pretrain_dt)))
    # set parameters
    parameters = {
        "stack_size": int(stack_size),
        "vocabulary_size": sentence_piece.vocab_size() + 1,
        "bert_dim_model": int(bert_dim_model),
        "multi_heads": int(head_size),
        "learning_rate": float(pt_lr),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "device": current_device,
        "corpus train size": len(pretrain_dt),
    }
    loss = nn.CrossEntropyLoss(ignore_index=sentence_piece.pad_id()).to(current_device)
    neptune.init('smeoni/bert-impl', api_token=neptune_api_token)
    neptune.create_experiment(name='bert_impl-experiment', params=parameters)
    # set the model
    bert = Bert(
        stack_size=parameters["stack_size"],
        voc_size=parameters["vocabulary_size"],
        dim_model=parameters["bert_dim_model"],
        mh_size=parameters["multi_heads"],
        padding_idx=sentence_piece.pad_id()
    ).to(current_device)
    parameters['model'] = bert
    parameters['optimizer'] = optim.Adam(bert.parameters(),
                                         lr=parameters['learning_rate'])
    parameters['loss'] = loss
    twitter_dt = TwitterDataset(pretrain_dt, sentence_piece)
    # pre train loop
    pre_train_loop(neptune, twitter_dt, checkpoint_path, train=True, **parameters)
