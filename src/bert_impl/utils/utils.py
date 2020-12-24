import concurrent.futures as futures
import glob
import os
import re
import random
from datetime import datetime

import numpy
import torch
from torch.utils.data import DataLoader

CLS = 'CLS'
MASK = 'MASK'
SEP = 'SEP'
PAD = 'PAD'
UNK = 'UNK'


def processing_text(entry, df_idx, spacy_nlp, column):
    text = entry[column].lower().replace("`", "'").strip()
    text = ' '.join([token.text for token in spacy_nlp(text) if not token.is_punct])
    text = re.sub(r'http[s]?://\S+', '[URL]', text).strip()
    return text, df_idx


def processing_df(dataframe, spacy_nlp):
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop=True)
    column = 'text'
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_text = {executor.submit(processing_text, df_entry, df_idx, spacy_nlp, column):
                           df_entry for df_idx, df_entry in enumerate(dataframe.iloc)}
        for future in futures.as_completed(future_text):
            res = future.result()
            dataframe.at[res[1], column] = res[0]
    dataframe = dataframe[dataframe[column] != '']
    dataframe = dataframe.reset_index(drop=True)
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
        for name, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data[name] = value.to(device)
            else:
                data[name] = value
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


def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def get_checkpoint_filename(id_fold, prefix="pt_", path="./"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path + "/bert_" + prefix + get_timestamp() + '_' + str(id_fold) + '.bin'


def create_sp_model(dataframe, path, spm):
    with open(path + '.txt', 'w') as voc_txt:
        for t_entry in dataframe['text']:
            voc_txt.write(t_entry + '\n')
        for sentiment in dataframe["sentiment"].unique().tolist():
            voc_txt.write(sentiment + '\n')
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
        .format(path, PAD, UNK, CLS, SEP)
    spm.SentencePieceTrainer.Train(spm_args)
    sentence_piece = spm.SentencePieceProcessor()

    return sentence_piece


def filter_selected_text(df_entry, df_idx, nlp, sentence_piece):
    pr_selected_text = processing_text(df_entry, df_idx, nlp, 'selected_text')
    selected_pieces = sentence_piece.EncodeAsPieces(pr_selected_text[0])
    return df_idx, [0 if piece not in selected_pieces else 1
                    for piece in sentence_piece.EncodeAsPieces(df_entry['text'])], pr_selected_text[
               0]


def filter_selected_text_df(dataframe, nlp, sentence_piece):
    dataframe["selected_vector"] = ''
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_text = {executor.submit(filter_selected_text, df_entry, df_idx, nlp, sentence_piece):
                           df_entry for df_idx, df_entry in enumerate(dataframe.iloc)}
        for future in futures.as_completed(future_text):
            res = future.result()
            dataframe.at[res[0], 'selected_text'] = res[2]
            dataframe.at[res[0], 'selected_vector'] = res[1]

    dataframe = dataframe[dataframe['selected_vector'] != '']
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def decode_sel_vector(word_emb_vector, dataset, select_vector):
    return dataset.sentence_piece.Decode(
        word_emb_vector[select_vector == 1].tolist())


def remove_checkpoints(checkpoint_path):
    files = glob.glob(checkpoint_path + "/*.bin")
    for file in files:
        os.remove(file)
