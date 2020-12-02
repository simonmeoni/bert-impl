from torch import nn


class NormalizeLayer(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, residual_in, prev_res):
        return residual_in + self.layer_norm(prev_res)
