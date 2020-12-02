import pandas as pd

import sentencepiece as spm
from src.bert_impl.dataset.bert_twitter_dataset import TwitterDataset
from src.bert_impl.utils.utils import UNK, generate_batches

df = pd.read_csv('./resources/test_data.csv')
sp = spm.SentencePieceProcessor()
sp.Load("./resources/test.model")
dataset = TwitterDataset(df.iloc[:4], df.iloc[4:6], df.iloc[6:9], sp)
expected_sentiment_list = [UNK, 'negative', 'neutral']


def test_switch_dataset():
    err = "the two int must have the same value"
    # test len of the train dataset
    assert len(dataset) == 4, err

    # test len of the eval dataset
    dataset.switch_to_dataset('eval')
    assert len(dataset) == 2, err

    # test len of the test dataset
    dataset.switch_to_dataset('test')
    assert len(dataset) == 3, err

    # test len of the test dataset
    try:
        dataset.switch_to_dataset('test1')
        assert False
    except ValueError:
        assert True


def test___init_sentiment_vocab():
    err = "these two lists must be equal"
    assert sorted(dataset.st_voc) == expected_sentiment_list, err


def test_get_sentiment_i():
    dataset.st_voc = expected_sentiment_list
    assert dataset.get_sentiment_i('negative') == 1
    assert dataset.get_sentiment_i('negative1') == 0


def test_vectorize():
    dataset_vectorize = TwitterDataset(df.iloc[0:1], df.iloc[0:1], df.iloc[0:1], sp)

    err = "the spacial marker must be contained in the tensor"

    # vector : [CLS, 'sooo', 'high', SEP, 'MASK']
    # noinspection PyArgumentList
    observed_v_1 = dataset_vectorize.vectorize("sooo high £¤").tolist()
    assert len(observed_v_1) == dataset_vectorize.max_seq_len, \
        "the tensor must have this size"
    assert 2 in observed_v_1, err
    assert 3 in observed_v_1, err
    assert 1 in observed_v_1, err
    assert 0 in observed_v_1, err


def test_generate_batches():
    test_dataset = TwitterDataset(df.iloc[0:10], df.iloc[0:10], df.iloc[0:10], sp)
    batch = next(generate_batches(test_dataset, 2))
    assert len(batch) == 2


def test_get_tokens():
    err = "these two lists must contains the same tokens"
    t_dataset = TwitterDataset(df.iloc[7:8], df.iloc[7:8], df.iloc[7:8], sp)
    t_dataset.max_seq_len = 3
    expected_sentence = "soooo high"
    vector = t_dataset.vectorize(expected_sentence)
    observed = t_dataset.get_tokens(vector)
    assert expected_sentence == observed, err
