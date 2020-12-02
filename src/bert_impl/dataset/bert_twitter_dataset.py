import torch
from torch.utils.data import Dataset

from src.bert_impl.utils.utils import UNK, MASK


class TwitterDataset(Dataset):
    def __init__(self, dataset, sentence_piece):
        self.dataset = dataset
        self.st_voc = []
        self.sentence_piece = sentence_piece
        self.max_seq_len = int(dataset['sequence length'].max()) + 2
        self.__init_sentiment_vocab()

    def __init_sentiment_vocab(self):
        self.st_voc = [UNK, *self.dataset['sentiment'].unique()]

    def get_vocab_size(self):
        return self.sentence_piece.vocab_size() + 1

    def __getitem__(self, index):
        return {
            'vectorized_tokens': self.vectorize(self.dataset.iloc[index]['text']),
            'sentiment_i': self.get_sentiment_i(self.dataset.iloc[index]['sentiment'])
        }

    def __len__(self):
        return len(self.dataset)

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
