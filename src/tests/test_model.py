import torch
from src.bert_implementation import AddNormalizeLayer, Encoder, \
    MultiHeadAttention, positional_enc, Bert


def test_compute_attention():
    err = "the size must be equal to this set"
    att = MultiHeadAttention(2, 4, 4)
    test_tensor = torch.rand(6, 4, 4)
    assert att.compute_attention(test_tensor, 6).shape == (6, 4, 2, 2), err


def test_multi_head_attention():
    err = "the size must be equal to this set"
    att = MultiHeadAttention(2, 4, 4)
    test_tensor = torch.rand(3, 2, 4)
    assert att(test_tensor).shape == (3, 2, 4), err


def test_add_normalize_layer():
    err = "the size must be equal to this set"
    add_norm = AddNormalizeLayer(4)
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
    observed = positional_enc(2, 4)
    assert torch.allclose(observed, expected), err


def test_encoder():
    err = "the size must be equal to this set"
    test_tensor = torch.rand(3, 2, 4)
    encoder = Encoder(3, 3, 4)
    assert encoder(test_tensor).shape == (3, 2, 4), err


def test_bert():
    err = "the size must be equal to this set"
    # noinspection PyArgumentList
    test_tensor = torch.LongTensor([
        [0, 1, 4],
        [3, 2, 1]
    ])
    bert = Bert(
        stack_size=6,
        embedding_dim=4,
        num_embeddings=5,
        dim_w_matrices=8,
        mh_size=4
    )
    assert bert(test_tensor).shape == (2, 3, 4), err
