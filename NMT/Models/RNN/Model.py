import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Utils.log import trace
from .Encoder import RNNEncoder
from .Decoder import RNNDecoder


class RNNModel(nn.Module):
    """
    Core RNN model for NMT.
    {
        RNN Encoder + RNN Decoder.
    }
    """
    def __init__(self, src_embedding, trg_embedding,
                 trg_vocab_size, config):
        super(RNNModel, self).__init__()

        self.encoder = RNNEncoder(
                            src_embedding,
                            config.rnn_type, 
                            config.src_embed_dim, 
                            config.hidden_size,
                            'decreasing' == config.mini_batch_sort_order, # True if decreasing, else false
                            config.enc_num_layers,
                            config.dropout, 
                            config.bidirectional)

        self.decoder = RNNDecoder(
                            trg_embedding,
                            config.rnn_type, 
                            config.trg_embed_dim, 
                            config.hidden_size,
                            config.dec_num_layers, 
                            config.attn_type,
                            config.bidirectional, 
                            config.dropout)
       
        self.generator = nn.Linear(config.hidden_size, trg_vocab_size)
        self.config = config
        if self.training:
            self.param_init()

    def param_init(self):
        trace("Initializing model parameters.")
        for p in self.parameters():
            p.data.uniform_(-0.1, 0.1)


    def translate_step(self, trg, encoder_outputs, lengths, decoder_state):
        
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        trg_embed = self.decoder.trg_embedding(trg)
        
        return self.decoder.forward_step(
            trg_embed, encoder_outputs, lengths, decoder_state)
    

    def forward(self, src, lengths, trg, state=None):
        """
        Forward propagate a `src` and `trg` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src     (Tensor)    : source sequence. [L_s x B]`.
            lengths (LongTensor): the src lengths, pre-padding `[B]`.
            trg     (LongTensor): source sequence. [L_t x B]`.
            state   (DecoderState): initial decoder state
        Returns:
            output `[trg_len x batch x hidden]`
            attention:  distributions of `[L_t x B x L_s]`
            state (DecoderState):     final decoder state
        """
        # encoding side
        encoder_outputs, encoder_state = \
                                self.encoder(src, lengths)

        # encoder to decoder
        decoder_state = \
            self.decoder.init_decoder_state(encoder_state)

        # decoding side
        decoder_outputs, decoder_state, attns = \
            self.decoder(trg, encoder_outputs, lengths, decoder_state)

        return decoder_outputs, decoder_state, attns



        # Basic attributes.
       

    