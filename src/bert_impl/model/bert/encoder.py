from torch import nn

from src.bert_impl.model.bert.ffnn import FeedForwardNetwork
from src.bert_impl.model.bert.mh_attention import MultiHeadAttention
from src.bert_impl.model.bert.norm_layer import NormalizeLayer


class Encoder(nn.Module):
    def __init__(self, dim_model, mh_size):
        super().__init__()
        self.mh_att = MultiHeadAttention(mh_size, dim_model)
        self.add_norm_l1 = NormalizeLayer(dim_model)
        self.feed_forward_network = FeedForwardNetwork(dim_model)
        self.add_norm_l2 = NormalizeLayer(dim_model)

    def forward(self, x_n, mask=None):
        z_n = self.mh_att(x_n, mask)
        l1_out = self.add_norm_l1(x_n, z_n)
        ffn_out = self.feed_forward_network(l1_out)
        return self.add_norm_l2(l1_out, ffn_out)
