import math

from torch import nn

from src.bert_impl.model.bert.pos_encoding import positional_enc
from src.bert_impl.model.bert.encoder import Encoder


class Bert(nn.Module):
    # pylint: disable=too-many-arguments
    def __init__(self, stack_size, voc_size,
                 dim_model, mh_size, padding_idx=0):
        super().__init__()
        self.dim_model = dim_model
        self.words_emb = nn.Embedding(
            embedding_dim=dim_model,
            num_embeddings=voc_size,
            padding_idx=padding_idx
        )
        self.sentence_emb = nn.Embedding(
            embedding_dim=dim_model,
            num_embeddings=voc_size,
            padding_idx=padding_idx
        )
        self.encoder_layer = nn.ModuleList()
        for _ in range(stack_size):
            self.encoder_layer.append(Encoder(dim_model, mh_size))

    def forward(self, word_tokens, sentence_tokens):
        mask = (word_tokens > 0).unsqueeze(1).repeat(1, word_tokens.size(1), 1).unsqueeze(1)
        embeddings = self.words_emb(word_tokens)
        sentence_tokens = self.sentence_emb(sentence_tokens)
        pos_embedding = positional_enc(embeddings.shape[1], embeddings.shape[2],
                                       self.words_emb.weight.device.type)
        z_n = pos_embedding + sentence_tokens + embeddings * math.sqrt(self.dim_model)
        for encoder in self.encoder_layer:
            z_n = encoder(z_n, mask)
        return z_n
