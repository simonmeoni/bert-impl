import pandas as pd
import torch

import sentencepiece as spm
from src.bert_impl.dataset.bert_twitter_dataset import TwitterDataset
from src.bert_impl.utils.utils import generate_masked_lm, replace_by_another_id, generate_batches, \
    generate_batched_masked_lm

df_test = pd.read_csv('resources/test_data.csv')
sp = spm.SentencePieceProcessor()
sp.Load("./resources/test.model")


def test_generate_masked_lm():
    t_dataset = TwitterDataset(df_test.iloc[0:1], sp)
    t_dataset.max_seq_len = 16
    # Check if mask function worked
    observed = generate_masked_lm(
        t_dataset[0]['words_embedding'],
        t_dataset,
        mask_prob=1,
        rnd_t_prob=0,
        unchanged_prob=0
    )
    assert not t_dataset[0]['words_embedding'].allclose(observed)
    assert len(observed) == t_dataset.max_seq_len + 5
    assert observed.unique(return_counts=True)[0].allclose(torch.LongTensor([0, 2, 3, 8000]))
    assert observed.unique(return_counts=True)[1].allclose(torch.LongTensor([6, 1,  2, 12]))

    # Check if unchanged prob has an effect
    observed = generate_masked_lm(
        t_dataset[0]['words_embedding'],
        t_dataset,
        mask_prob=1.,
        rnd_t_prob=0,
        unchanged_prob=1.
    )
    assert t_dataset[0]['words_embedding'].allclose(observed)


def test_replace_by_another_token():
    t_dataset = TwitterDataset(df_test.iloc[0:1], sp)

    replaced_ids = [replace_by_another_id(4, t_dataset) for _ in range(30)]
    replaced_tokens = t_dataset.get_tokens(torch.LongTensor(replaced_ids))

    assert 4 not in replaced_ids
    assert t_dataset.get_pad() not in replaced_ids
    assert t_dataset.get_mask() not in replaced_ids
    assert t_dataset.get_cls() not in replaced_ids
    assert t_dataset.get_sep() not in replaced_ids
    assert replaced_tokens
    print(replaced_tokens)


def test_generate_batched_masked_lm():
    t_dataset = TwitterDataset(df_test.iloc[0:10], sp)
    batch = next(generate_batches(t_dataset, 4))
    batch_masked_lm = generate_batched_masked_lm(batch['words_embedding'], t_dataset)
    assert batch['words_embedding'].shape == batch_masked_lm.shape
