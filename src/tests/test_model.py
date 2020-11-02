import torch
from src.bert_implementation import AddNormalizeLayer, Attention, Encoder, \
                                    MultiHeadAttention, positional_enc, Bert

def test_attention():
    err = "the size must be equal to this set"
    att = Attention(4,3)
    test_tensor = torch.rand(3,2,4)
    assert att(test_tensor).shape == (3,2,3), err

def test_multi_head_attention():
    err = "the size must be equal to this set"
    att = MultiHeadAttention(8,4,3)
    test_tensor = torch.rand(3,2,4)
    assert att(test_tensor).shape == (3,2,4), err

def test_add_normalize_layer():
    err = "the size must be equal to this set"
    add_norm = AddNormalizeLayer(4)
    test_zn = torch.rand(3,2,4)
    test_xn = torch.rand(3,2,4)
    assert add_norm(test_xn, test_zn).shape == (3,2,4), err

def test_positional_enc():
    err = "these lists must be equal"
    expected = torch.Tensor([
        [0, 1.0000e+00, 0.0000e+00, 1.0000e+00],
        [1.0000e-04, 1.0000e+00, 9.9990e-05, 1.0000e+00]
    ])
    observed = positional_enc(2,4)
    assert torch.allclose(observed,expected), err

def test_encoder():
    err = "the size must be equal to this set"
    test_tensor = torch.rand(3,2,4)
    encoder = Encoder(3,3,4)
    assert encoder(test_tensor).shape == (3,2,4), err


def test_bert():
    err = "the size must be equal to this set"
    test_tensor = torch.rand(3,2,4)
    bert = Bert(
        stack_size=2,
        embedding_dim=4,
        num_embeddings=4,
        dim_w_matrices=3,
        mh_size=6
        )
    assert bert(test_tensor).shape == (3,2,4), err
