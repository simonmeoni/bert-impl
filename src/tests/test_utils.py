import sentencepiece as spm
import pandas as pd
import spacy
import torch

from src.bert_impl.utils.utils import extract_selected_text, processing_df, collate_fn


def test_extract_selected_text():
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
    df_test = processing_df(pd.read_csv('resources/test_selected_text_data.csv'), nlp)
    sentence_piece = spm.SentencePieceProcessor()
    sentence_piece.Load("./resources/test.model")

    expected_vector_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert extract_selected_text(df_test.loc[4], 4, nlp, sentence_piece)[1] == expected_vector_1
    expected_vector_2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert extract_selected_text(df_test.loc[6], 6, nlp, sentence_piece)[1] == expected_vector_2


def test_collate_fn():
    test_batch = torch.load("resources/test_batch_with_selected_text.pickle")
    collate_test_batch = collate_fn(test_batch)
    len_collate = set(len(v) for i in collate_test_batch.values()
                      for v in i if isinstance(v, torch.Tensor))
    expected_len = 41
    assert len(len_collate) == 1
    assert expected_len == len_collate.pop()

    test_batch = torch.load("resources/test_batch.pickle")
    collate_test_batch = collate_fn(test_batch)
    len_collate = set(len(v) for i in collate_test_batch.values()
                      for v in i if isinstance(v, torch.Tensor))
    expected_len = 41
    assert len(len_collate) == 1
    assert expected_len == len_collate.pop()
