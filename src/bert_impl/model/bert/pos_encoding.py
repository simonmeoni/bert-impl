import math

import torch


def positional_enc(seq_len, model_dim, device="cpu"):
    pos_emb_vector = torch.empty(seq_len, model_dim).to(device)
    for pos in range(seq_len):
        for i_col in range(model_dim):
            power_ind = 10000 ** ((2 * i_col) / model_dim)
            if i_col % 2 == 0:
                pos_emb_vector[pos, i_col] = math.sin(pos / power_ind)
            else:
                pos_emb_vector[pos, i_col] = math.cos(pos / power_ind)
    return pos_emb_vector
