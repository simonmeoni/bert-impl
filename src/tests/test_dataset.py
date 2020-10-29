import pandas as pd
from src.bert_implementation import TwitterDataset

df = pd.read_csv('src/tests/resources/test_data.csv')
dataset = TwitterDataset(df.iloc[:4], df.iloc[4:6], df.iloc[6:9])
expected_sentiment_list = ['UNK', 'negative', 'neutral']

def test_switch_dataset():
    # test len of the train dataset
    assert len(dataset) == 4

    # test len of the eval dataset
    dataset.switch_to_dataset('eval')
    assert len(dataset) == 2

    # test len of the test dataset
    dataset.switch_to_dataset('test')
    assert len(dataset) == 3

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
    dataset___init_vocab = TwitterDataset(df.iloc[2:4], [], [])
    err = "these vocabulary must be equals"
    expected_voc = [ 'UNK', 'SOS', 'EOS', 'MASK', '-PRON-', 'boss', 'be', 'bully',
                        '...', 'what', 'interview', '!', 'leave', 'alone']
    assert sorted(dataset___init_vocab.vocabulary['tokens']) == sorted(expected_voc), err
    assert dataset___init_vocab.vocabulary['max_seq_len'] == 8

def test_vectorize():
    dataset_vectorize = TwitterDataset(df.iloc[0:0], [], [])
    dataset_vectorize.vocabulary['tokens'] = ['of', 'SOS', 'MASK', 'high',
                                                'both', 'EOS', 'sooo', '-PRON-', 'UNK']
    dataset_vectorize.vocabulary['max_seq_len'] = 5

    err = "these vectors must be equals"

    # vector : ['SOS', 'sooo', 'high', 'EOS', 'MASK']
    expected_v_1 = [1, 6, 3, 5, 2]
    # vector : ['SOS', 'both', 'of', 'you', 'EOS']
    expected_v_2 = [1, 4, 0, 7, 5]
    observed_v_1 = dataset_vectorize.vectorize("Sooo high")
    observed_v_2 = dataset_vectorize.vectorize("Both of you")

    assert len(observed_v_1) == len(observed_v_2)
    assert expected_v_1 == observed_v_1, err
    assert expected_v_2 == observed_v_2, err

def test__get_item__():
    pass
