import torch
import torch.nn as nn
import math
from torch.autograd import Variable

MAX_SIZE = 5000

class PositionalEncoding(nn.Module):
    """
    Args:
       dropout (float): 
       dim (int): embedding size
    """
    def __init__(self, dropout, dim, max_len=MAX_SIZE):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        
        emb = emb * math.sqrt(self.dim)
        emb = emb + Variable(self.pe[:emb.size(0)], requires_grad=False)
        emb = self.dropout(emb)
        return emb
