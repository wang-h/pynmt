import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .Modules import StackedCNN

class CNNEncoder(nn.Module):
    def __init__(self, src_embedding, embed_dim, 
                    hidden_size, num_layers, 
                    dropout=0.2, kernel_width=3):
        super(CNNEncoder, self).__init__()
        self.embedding = src_embedding
        self.bottle = nn.Linear(embed_dim, hidden_size)
        self.cnn = StackedCNN(hidden_size, num_layers, 
                              kernel_width, dropout)

    def forward(self, input, lengths=None):
        emb = self.bottle(self.embeddings(input.transpose(0, 1)))
        output = self.cnn(emb.unsqueeze(3))
        output = output.squeeze(3).transpose(0, 1)
        return output, emb.transpose(0, 1)
