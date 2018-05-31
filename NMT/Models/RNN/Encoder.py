import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class RNNEncoder(nn.Module):
    def __init__(self, 
                src_embedding, 
                rnn_type, 
                embed_dim,
                hidden_size, 
                num_layers=2, 
                dropout=0.0, 
                bidirectional=True):

        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type)(embed_dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        self.hidden_size = hidden_size * num_directions
        self.src_embedding = src_embedding

    def fix_final_state_size(self, final_state):
        def resize(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], -1)
            return h
        if self.bidirectional:
            if self.rnn_type == "GRU":
                final_state = resize(final_state)
            elif self.rnn_type == "LSTM":
                final_state = tuple(resize(h) for h in final_state)
        return final_state

    def forward(self, src, lengths=None):
        """
        Args:
            src (`LongTensor`): sequences of padded tokens `[L_s x B]`. 
            lengths (`LongTensor`): the padded source lengths `[B]`.
        Returns:
            (`FloatTensor`,:obj:`nmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[L_t x B x H]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over source words at each target word
                        `[L_t x B x L_s]`.
        """
        src_embed = self.src_embedding(src)
        if lengths is not None:
            packed = pack(src_embed, lengths.view(-1).tolist())    
        
        output, final_state = self.rnn(packed)

        if lengths is not None:
            output = unpack(output)[0]

        final_state = self.fix_final_state_size(final_state)
        
        return output, final_state 