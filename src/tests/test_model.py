import torch

from src.bert_impl.model.bert.bert import Bert
from src.bert_impl.model.bert.encoder import Encoder
from src.bert_impl.model.bert.mh_attention import MultiHeadAttention
from src.bert_impl.model.bert.norm_layer import NormalizeLayer
from src.bert_impl.model.bert.pos_encoding import positional_enc


def test_compute_attention():
    err = "the size must be equal to this set"
    att = MultiHeadAttention(2, 4)
    test_tensor = torch.rand(6, 4, 4)
    assert att.compute_attention(test_tensor, 6).shape == (6, 2, 4, 2), err


def test_multi_head_attention():
    err = "the size must be equal to this set"
    att = MultiHeadAttention(2, 4)
    test_tensor = torch.rand(3, 2, 4)
    assert att(test_tensor).shape == (3, 2, 4), err


def test_add_normalize_layer():
    err = "the size must be equal to this set"
    add_norm = NormalizeLayer(4)
    test_zn = torch.rand(3, 2, 4)
    test_xn = torch.rand(3, 2, 4)
    assert add_norm(test_xn, test_zn).shape == (3, 2, 4), err


def test_positional_enc():
    err = "these lists must be equal"
    # noinspection PyArgumentList
    expected = torch.Tensor([
        [0, 1.0000e+00, 0.0000e+00, 1.0000e+00],
        [8.4147e-01, 9.9995e-01, 1.0000e-04, 1.0000e+00]
    ])
    observed = positional_enc(2, 4, device="cpu")
    assert torch.allclose(observed, expected), err


def test_encoder():
    err = "the size must be equal to this set"
    test_tensor = torch.rand(3, 2, 4)
    encoder = Encoder(4, 4)
    assert encoder(test_tensor).shape == (3, 2, 4), err


def test_bert():
    err = "the size must be equal to this set"
    # noinspection PyArgumentList
    test_tensor = torch.LongTensor([
        [0, 1, 4],
        [3, 2, 1]
    ]).to("cpu")
    bert = Bert(
        stack_size=6,
        voc_size=5,
        dim_model=4,
        mh_size=4
    ).to("cpu")
    assert bert(test_tensor, test_tensor).shape == (2, 3, 4), err
