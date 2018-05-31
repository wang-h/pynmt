import math
import torch
import torch.nn as nn
from torch.autograd import Variable
class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    """

    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.pw_ff1 = nn.Sequential(
            nn.LayerNorm(size, eps=1e-06),
            nn.Linear(size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pw_ff2 = nn.Sequential(
            nn.Linear(hidden_size, size),
            nn.Dropout(dropout))

    def forward(self, x):
        inter = self.pw_ff1(x)
        output = self.pw_ff2(inter)
        return output + x