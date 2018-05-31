import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from .Modules import MultiHeadedAttention
from .Modules import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    """

    def __init__(self, size, dropout, num_heads=8, hidden_size=1024):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            num_heads, size, dropout=dropout)
        self.ff = PositionwiseFeedForward(
            size, hidden_size, dropout)
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask):
        input_norm = self.layer_norm(input)

        context, _ = self.self_attn(
            input_norm, input_norm, input_norm,mask=mask)

        # residual
        out = self.dropout(context) + input
        return self.ff(out)

class TransformerEncoder(nn.Module):
    """
    Args:
       num_layers (int): number of encoder layers
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    """
    def __init__(self, 
                src_embedding,
                embed_dim,
                hidden_size, 
                inner_hidden_size,
                num_layers=2, 
                dropout=0.0, 
                num_heads=8,
                padding_idx=0):

        super(TransformerEncoder, self).__init__()
        self.src_embedding = src_embedding
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.inner_hidden_size = inner_hidden_size
        self.transformer = \
                    nn.ModuleList(
                        [
                            TransformerEncoderLayer(embed_dim, dropout, num_heads, inner_hidden_size)\
                            for i in range(num_layers)
                        ])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def make_mask(self, src):
        words = src.transpose(0, 1)
        mask = words.data.eq(self.padding_idx)
        mask = mask.unsqueeze(1).repeat(1, words.size(-1), 1) 
        return mask
         
    def forward(self, src, lengths=None):
        src_embed = self.src_embedding(src)
        src_mask = self.make_mask(src)
        
        # Run the forward pass of every layer of the tranformer.
        output = src_embed.transpose(0, 1)
        for i in range(self.num_layers):
            output = self.transformer[i](output, src_mask)
        output = self.layer_norm(output)
        return output.transpose(0, 1), src