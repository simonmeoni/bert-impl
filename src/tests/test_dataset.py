import pandas as pd
import sentencepiece as spm

from src.bert_impl.dataset.bert_twitter_dataset import TwitterDataset
from src.bert_impl.utils.utils import generate_batches

df = pd.read_csv('./resources/test_data.csv')
sp = spm.SentencePieceProcessor()
sp.Load("./resources/test.model")
dataset = TwitterDataset(df.iloc[:4], sp)
expected_sentiment_list = ['negative', 'neutral']


def test___init_sentiment_vocab():
    err = "these two lists must be equal"
    assert sorted(dataset.st_voc) == expected_sentiment_list, err


def test_vectorize():
    dataset_vectorize = TwitterDataset(df.iloc[0:1], sp)
    # vector : [CLS, 'sooo', 'high', SEP, 'MASK']
    # noinspection PyArgumentList
    observed_v_1 = dataset_vectorize.vectorize("sooo high £¤", "neutral")
    assert dataset.sentence_piece.decode(observed_v_1[0].tolist()) == 'neutral sooo high  ⁇ '


def test_generate_batches():
    test_dataset = TwitterDataset(df.iloc[0:10], sp)
    print(test_dataset.max_seq_len)
    batch = next(generate_batches(test_dataset, 10))
    assert len(batch) == 5


def test_get_tokens():
    err = "these two lists must contains the same tokens"
    t_dataset = TwitterDataset(df.iloc[7:8], sp)
    t_dataset.max_seq_len = 3
    expected_sentence = "soooo high"
    vector = t_dataset.vectorize(expected_sentence, 'neutral')
    observed = t_dataset.get_tokens(vector[0])
    assert 'neutral  ' + expected_sentence == observed, err
