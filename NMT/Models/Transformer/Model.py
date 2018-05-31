import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_

from .Encoder import TransformerEncoder
from .Decoder import TransformerDecoder
from .DecoderState import TransformerDecoderState
from .Modules import PositionalEncoding

# class BaseTransformerModel(nn.Module):
#     """
#     Core RNN model for NMT.
#     {
#         Transformer Encoder + Transformer Decoder.
#     }
#     """
#     def __init__(self, src_embedding, trg_embedding,
#                  trg_vocab_size, padding_idx, config):
#        super(BaseTransformerModel, self).__init__()
class TransformerModel(nn.Module):
    """
    Core RNN model for NMT.
    {
        Transformer Encoder + Transformer Decoder.
    }
    """
    def __init__(self, src_embedding, trg_embedding,
                 trg_vocab_size, padding_idx, config):
        super(TransformerModel, self).__init__()
        self.padding_idx = padding_idx

        self.src_embedding = nn.Sequential(
            src_embedding,    
            PositionalEncoding(config.dropout, config.src_embed_dim)
        )
        
        self.trg_embedding = nn.Sequential(
            trg_embedding,    
            PositionalEncoding(config.dropout, config.trg_embed_dim)
        )

        self.encoder = TransformerEncoder(
                            self.src_embedding,
                            config.src_embed_dim, 
                            config.hidden_size, 
                            config.inner_hidden_size,
                            num_layers=config.enc_num_layers,
                            dropout=config.dropout,
                            num_heads=config.num_heads,
                            padding_idx=padding_idx)


        self.decoder = TransformerDecoder(
                            self.trg_embedding, 
                            config.trg_embed_dim, 
                            config.hidden_size,
                            config.inner_hidden_size,
                            num_layers=config.dec_num_layers, 
                            attn_type=config.attn_type,
                            dropout=config.dropout,
                            num_heads=config.num_heads,
                            padding_idx=0)
      
  
        
        self.generator = nn.Linear(config.hidden_size, trg_vocab_size)
        self.config = config
        if self.training:
            self.param_init()

    def param_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def make_mask(self, input):
        words = input.transpose(0, 1)
        mask = words.data.eq(self.padding_idx)
        mask = mask.unsqueeze(1).repeat(1, words.size(-1), 1) 
        return mask



    def decode(self, trg, encoder_outputs, src_lengths, decoder_state):
        
        return self.decoder(trg, encoder_outputs,
                         src_lengths, decoder_state)

    def translate_step(self, trg, encoder_outputs, lengths, decoder_state):
        output, state, attn = \
            self.decoder(trg, encoder_outputs, lengths, decoder_state)
        #print(output.size(), attn.size())
        return output.squeeze(0), attn.squeeze(0), state

    def forward(self, src, src_lengths, trg, decoder_state=None):
        # encoding side
        encoder_outputs, src = self.encoder(src, src_lengths)

        # encoder to decoder
        if decoder_state is None:
            decoder_state = self.decoder.init_decoder_state(src)

        # decoding side
        decoder_outputs, decoder_state, attns = \
            self.decoder(trg, encoder_outputs, src_lengths, decoder_state)
        return decoder_outputs, attns, decoder_state
