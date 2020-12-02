import concurrent.futures as futures
import os
import re
import random

import numpy
import torch
from torch.utils.data import DataLoader

CLS = 'CLS'
MASK = 'MASK'
SEP = 'SEP'
PAD = 'PAD'
UNK = 'UNK'

TRAIN_PATH = '../input/tweet-sentiment-extraction/train.csv'
TEST_PATH = '../input/tweet-sentiment-extraction/test.csv'
PR_TRAIN_PATH = './src/resources/processed_train.csv'
PR_TEST_PATH = './src/resources/processed_test.csv'

NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")


def processing_text(entry, dataframe, df_idx, spacy_nlp):
    text = entry['text'].lower().replace("`", "'").strip()
    text = ' '.join([token.text
                     if token.lemma_ == "-PRON-" or '*' in token.text else token.lemma_
                     if not token.is_punct else '' for token in spacy_nlp(text)]).strip()
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    dataframe.at[df_idx, 'text'] = re.sub(r'\s\s+', ' ', text)


def processing_df(dataframe, path, spacy_nlp):
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop=True)

    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(processing_text, df_entry, dataframe, df_idx, spacy_nlp):
                             df_entry for df_idx, df_entry in enumerate(dataframe.iloc)}
    for _ in futures.as_completed(future_to_url):
        pass
    dataframe = dataframe[dataframe['text'] != '']
    dataframe = dataframe.reset_index(drop=True)
    dataframe.to_csv(path)
    return dataframe


def generate_masked_lm(vector, dataset, mask_prob=.15, rnd_t_prob=.1, unchanged_prob=.1):
    return torch.LongTensor([
        replace_token(idx_token, dataset, rnd_t_prob, unchanged_prob)
        if numpy.random.uniform() < mask_prob and is_not_markers(idx_token, dataset)
        else idx_token
        for idx_token in vector
    ])


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


def generate_batched_masked_lm(batched_vectors, dataset,
                               mask_prob=.15, rnd_t_prob=.1, unchanged_prob=.1):
    batched_masked_lm = [
        generate_masked_lm(vector, dataset, mask_prob, rnd_t_prob, unchanged_prob)
        for vector in batched_vectors
    ]
    return torch.stack(batched_masked_lm)


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
        replaced_index_t = random.choice(range(dataset.get_vocab_size()))
    return replaced_index_t


def set_seq_length(dataframe, sentence_piece):
    dataframe['sequence length'] = ''
    for idx, entry in enumerate(dataframe.iloc):
        dataframe.at[idx, 'sequence length'] = len(sentence_piece.EncodeAsIds(entry['text']))
    print('number of entries in the dataset : ' + str(len(dataframe)))
