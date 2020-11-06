import pandas as pd
from src.bert_implementation import TwitterDataset, generate_batched_masked_lm, \
    generate_batches, generate_masked_lm, replace_by_another_token, CLS, SEP, MASK

df_test = pd.read_csv('src/tests/resources/test_data.csv')


def test_generate_masked_lm():
    t_dataset = TwitterDataset(df_test.iloc[0:1], df_test.iloc[0:1], df_test.iloc[0:1])
    t_dataset.vocabulary['max_seq_len'] = 15
    # Check if mask function worked
    observed = generate_masked_lm(
        t_dataset[0]['vectorized_tokens'],
        t_dataset,
        mask_prob=1,
        rnd_t_prob=0,
        unchanged_prob=0
    )
    assert not t_dataset[0]['vectorized_tokens'].allclose(observed)
    assert t_dataset.get_tokens(observed).count(CLS) == 1
    assert t_dataset.get_tokens(observed).count(SEP) == 1
    assert t_dataset.get_tokens(observed).count(MASK) >= 5

    # Check if unchanged prob has an effect
    observed = generate_masked_lm(
        t_dataset[0]['vectorized_tokens'],
        t_dataset,
        mask_prob=1.,
        rnd_t_prob=0,
        unchanged_prob=1.
    )
    assert t_dataset[0]['vectorized_tokens'].allclose(observed)


def test_replace_by_another_token():
    t_dataset = TwitterDataset(df_test.iloc[0:1], df_test.iloc[0:1], df_test.iloc[0:1])
    t_dataset.vocabulary['tokens'] = [CLS, SEP, MASK, 'token_1', 'token_2']
    replaced_tokens = [replace_by_another_token(3, t_dataset) for _ in range(30)]
    assert replaced_tokens == [4] * 30


def test_generate_batched_masked_lm():
    t_dataset = TwitterDataset(df_test.iloc[0:10], df_test.iloc[0:10], df_test.iloc[0:10])
    batch = next(generate_batches(t_dataset, 4))
    batch_masked_lm = generate_batched_masked_lm(batch['vectorized_tokens'], t_dataset)
    assert batch['vectorized_tokens'].shape == batch_masked_lm.shape
