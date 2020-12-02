import torch
from torch.utils.data import Dataset
import pandas as pd

from src.bert_impl.utils.utils import UNK, MASK


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
            'vectorized_tokens': self.vectorize(self.current_dataset.iloc[index]['text']),
            'sentiment_i': self.get_sentiment_i(self.current_dataset.iloc[index]['sentiment'])
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
            raise ValueError('this dataset doesn\'t exist !')

    # noinspection PyArgumentList
    def vectorize(self, tokens):
        vector = self.sentence_piece.EncodeAsIds(tokens)
        return torch.LongTensor(
            [self.sentence_piece.bos_id()] + vector + [self.sentence_piece.eos_id()] +
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
