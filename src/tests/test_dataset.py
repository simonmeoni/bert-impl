import pandas as pd
import torch
from src.bert_implementation import TwitterDataset, generate_batches, UNK, CLS, SEP

df = pd.read_csv('src/tests/resources/test_data.csv')
dataset = TwitterDataset(df.iloc[:4], df.iloc[4:6], df.iloc[6:9])
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


def test___init_vocab():
    dataset___init_vocab = TwitterDataset(df.iloc[2:4], df.iloc[2:4], df.iloc[2:4])
    err = "these vocabulary must be equals"
    expected_voc = [UNK, CLS, SEP, 'MASK', '-PRON-', 'boss', 'be', 'bully',
                    '...', 'what', 'interview', '!', 'leave', 'alone']
    assert sorted(dataset___init_vocab.vocabulary['tokens']) == sorted(expected_voc), err
    assert dataset___init_vocab.vocabulary['max_seq_len'] == 8


def test_vectorize():
    dataset_vectorize = TwitterDataset(df.iloc[0:0], df.iloc[0:0], df.iloc[0:0])
    dataset_vectorize.vocabulary['tokens'] = ['of', CLS, 'MASK', 'high',
                                              'both', SEP, 'sooo', '-PRON-', UNK]
    dataset_vectorize.vocabulary['max_seq_len'] = 5

    err = "these vectors must be equals"

    # vector : [CLS, 'sooo', 'high', SEP, 'MASK']
    expected_v_1 = torch.LongTensor([1, 6, 3, 5, 2])
    # vector : [CLS, 'both', 'of', 'you', SEP]
    expected_v_2 = torch.LongTensor([1, 4, 0, 7, 5])
    # vector : [CLS, UNK, UNK, UNK, SEP]
    expected_v_3 = torch.LongTensor([1, 8, 8, 8, 5])
    observed_v_1 = dataset_vectorize.vectorize("Sooo high")
    observed_v_2 = dataset_vectorize.vectorize("Both of you")
    observed_v_3 = dataset_vectorize.vectorize("Cheese Fromage Renard")

    assert len(observed_v_1) == len(observed_v_2)
    assert expected_v_1.allclose(observed_v_1), err
    assert expected_v_2.allclose(observed_v_2), err
    assert expected_v_3.allclose(observed_v_3), err


def test_generate_batches():
    test_dataset = TwitterDataset(df.iloc[0:10], df.iloc[0:10], df.iloc[0:10])
    assert next(generate_batches(test_dataset, 2))


def test_get_tokens():
    err = "these two lists must contains the same tokens"
    t_dataset = TwitterDataset(df.iloc[7:8], df.iloc[7:8], df.iloc[7:8])
    expected = [CLS, 'soooo', 'high', SEP]
    observed = t_dataset.get_tokens(t_dataset[0]['vectorized_tokens'])
    assert expected == observed, err
