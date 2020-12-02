from torch import nn
import torch.nn.functional as f


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_model * 4)
        self.linear_2 = nn.Linear(dim_model * 4, dim_model)

    def forward(self, x_n):
        out_l1 = f.relu(self.linear_1(x_n))
        return self.linear_2(out_l1)
