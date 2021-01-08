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
from src.bert_impl.train.loop import fine_tuning_loop
from src.bert_impl.utils.utils import processing_df, \
    set_seq_length, get_checkpoint_filename, filter_selected_text_df


# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def fine_tune_bert_model(train_path,
                         ft_train_path,
                         pretrain_model_path,
                         save_model_path,
                         sp_path,
                         neptune_api_token,
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

    sentence_piece = spm.SentencePieceProcessor()
    sentence_piece.Load(sp_path + '.model')

    # initialize corpus

    if Path(ft_train_path).is_file():
        train_csv = pd.read_pickle(ft_train_path)
        train_csv.astype(object)
    else:
        train_csv = pd.read_csv(train_path, dtype={'text': 'string'})
        nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        train_csv = processing_df(train_csv, nlp)
        train_csv = train_csv[train_csv['sentiment'] != 'neutral']
        train_csv = train_csv.reset_index(drop=True)
        train_csv = filter_selected_text_df(train_csv, nlp, sentence_piece)
        train_csv.to_pickle(ft_train_path)
    # Set the corpus size
    if int(corpus_size) >= 0:
        train_csv = train_csv[:int(corpus_size)]
    # set the length of the different entries and remove certain cases
    set_seq_length(train_csv, sentence_piece)
    print("size of the dataset : {0}".format(len(train_csv)))
    # set parameters
    parameters = {
        "vocabulary_size": sentence_piece.vocab_size() + 1,
        "st_learning_rate": float(st_lr),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "device": current_device,
        "corpus train size": len(train_csv),
        "folds": int(folds)
    }
    loss = nn.CrossEntropyLoss().to(current_device)
    neptune.init('smeoni/bert-impl', api_token=neptune_api_token)
    neptune.create_experiment(name='bert_impl-experiment', params=parameters)
    folds = KFold(n_splits=parameters['folds'], shuffle=False)
    cv_ft_loss = []
    cv_ft_jaccard = []
    loaded_model = torch.load(pretrain_model_path)
    for id_fold, fold in enumerate(folds.split(train_csv)):
        # set the model
        bert = parameters['model'] = loaded_model['model'].to(current_device)
        parameters['stack_size'] = loaded_model['stack_size']
        parameters["bert_dim_model"] = loaded_model['bert_dim_model']
        parameters["multi_heads"] = loaded_model["multi_heads"]
        parameters['st_optimizer'] = optim.Adam(bert.parameters(),
                                                lr=parameters['st_learning_rate'])
        parameters['loss'] = loss
        train_fold, eval_fold = fold
        train_dt = TwitterDataset(train_csv.iloc[train_fold], sentence_piece)
        eval_dt = TwitterDataset(train_csv.iloc[eval_fold], sentence_piece)
        # fine tuning loop
        fine_tuning_loop(neptune, train_dt, True, **parameters)
        cv_score_ft, jaccard_score = fine_tuning_loop(neptune, eval_dt, train=False, **parameters)
        # logging and metrics
        torch.save(bert, get_checkpoint_filename(prefix="ft_", id_fold=id_fold,
                                                 path=save_model_path))
        neptune.log_metric('fine tuning loss cross validation', cv_score_ft)
        neptune.log_metric('fine tuning jaccard score cross validation', jaccard_score)
        cv_ft_loss.append(cv_score_ft)
        cv_ft_jaccard.append(jaccard_score)
    mean_cv_ft_loss = mean(cv_ft_loss)
    mean_cv_ft_jaccard = mean(cv_ft_jaccard)
    print("""
loss cross validation score mean :
* fine-tuning :  {0}
loss cross validation scores :
* fine-tuning :  {1}
jaccard cross validation score mean :
* fine-tuning :  {2}
jaccard cross validation scores :
* fine-tuning :  {3}
    """.format(mean_cv_ft_loss, cv_ft_loss, mean_cv_ft_jaccard, cv_ft_jaccard))

    neptune.log_metric('mean fine tuning cross validation', mean_cv_ft_loss)
