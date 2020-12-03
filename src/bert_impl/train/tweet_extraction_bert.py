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
from src.bert_impl.utils.utils import processing_df, \
    PAD, UNK, \
    CLS, SEP, set_seq_length, get_checkpoint_filename


# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def train_bert_model(train_path,
                     pr_train_path,
                     checkpoint_path,
                     sp_path,
                     neptune_api_token,
                     stack_size,
                     bert_dim_model,
                     head_size,
                     pt_lr,
                     st_lr,
                     batch_size,
                     epochs,
                     folds,
                     corpus_size):
    # set to cuda device if it is available
    if torch.cuda.is_available():
        torch_device = "cuda"
    else:
        torch_device = "cpu"
    current_device = torch.device(torch_device)
    # initialize corpus
    if not Path(pr_train_path).is_file():
        train_csv = pd.read_csv(train_path, dtype={'text': 'string'})
    else:
        train_csv = pd.read_csv(pr_train_path, dtype={'text': 'string'})
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
    if not Path(pr_train_path).is_file():
        train_csv = processing_df(train_csv, pr_train_path, nlp)
    with open(sp_path + '.txt', 'w') as voc_txt:
        for t_entry in train_csv['text']:
            voc_txt.write(t_entry + '\n')
    # initialize sentence piece
    spm_args = "" \
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
        .format(sp_path, PAD, UNK, CLS, SEP)
    spm.SentencePieceTrainer.Train(spm_args)
    sentence_piece = spm.SentencePieceProcessor()
    sentence_piece.Load(sp_path + '.model')
    # Set the corpus size
    if int(corpus_size) >= 0:
        train_csv = train_csv[:int(corpus_size)]
    # set the length of the different entries and remove certain cases
    set_seq_length(train_csv, sentence_piece)
    train_csv = train_csv.drop(train_csv[train_csv['sequence length'].ge(35)].index)
    train_csv = train_csv.drop(train_csv[train_csv['sequence length'].le(5)].index)
    train_csv = train_csv.reset_index(drop=True)
    print("size of the dataset : {0}".format(len(train_csv)))
    # set parameters
    parameters = {
        "stack_size": int(stack_size),
        "vocabulary_size": sentence_piece.vocab_size() + 1,
        "bert_dim_model": int(bert_dim_model),
        "multi_heads": int(head_size),
        "learning_rate": float(pt_lr),
        "st_learning_rate": float(st_lr),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "device": current_device,
        "corpus train size": len(train_csv),
        "folds": int(folds)
    }
    loss = nn.CrossEntropyLoss(ignore_index=sentence_piece.pad_id()) \
        .to(current_device)
    neptune.init('smeoni/bert-impl', api_token=neptune_api_token)
    neptune.create_experiment(name='bert_impl-experiment', params=parameters)
    folds = KFold(n_splits=parameters['folds'], shuffle=False)
    cv_pt = []
    cv_ft = []
    for id_fold, fold in enumerate(folds.split(train_csv)):
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
        parameters['st_optimizer'] = optim.Adam(bert.parameters(),
                                                lr=parameters['st_learning_rate'])
        parameters['loss'] = loss

        train_fold, eval_fold = fold
        train_dt = TwitterDataset(train_csv.iloc[train_fold], sentence_piece)
        eval_dt = TwitterDataset(train_csv.iloc[eval_fold], sentence_piece)
        # pre train loop
        pre_train_loop(neptune, train_dt, train=True, **parameters)
        cv_score_pt = pre_train_loop(neptune, eval_dt, train=False, **parameters)
        torch.save(bert, get_checkpoint_filename(id_fold=id_fold, path=checkpoint_path))
        # fine tuning loop
        fine_tuning_loop(neptune, train_dt, True, **parameters)
        cv_score_ft = fine_tuning_loop(neptune, eval_dt, train=False, **parameters)
        torch.save(bert, get_checkpoint_filename(prefix="ft_", id_fold=id_fold,
                                                 path=checkpoint_path))
        # logging and metrics
        neptune.log_metric('pre-training cross validation', cv_score_pt)
        neptune.log_metric('fine tuning cross validation', cv_score_ft)
        cv_pt.append(cv_score_pt)
        cv_ft.append(cv_score_ft)
    m_cv_pt = mean(cv_pt)
    m_cv_ft = mean(cv_ft)
    print("""
cross validation score mean :
* pre-training : {0}
* fine-tuning :  {1}
cross validation scores : 
* pre-training : {2}
* fine-tuning :  {3}    
    """.format(m_cv_pt, m_cv_ft, cv_pt, cv_ft))

    neptune.log_metric('mean fine tuning cross validation', m_cv_ft)
    neptune.log_metric('mean fine tuning cross validation', m_cv_pt)
