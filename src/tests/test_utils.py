import sentencepiece as spm
import pandas as pd
import spacy
from src.bert_impl.utils.utils import extract_selected_text, processing_df


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
