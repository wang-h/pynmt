import torch
import torch.nn as nn
from NMT.Modules import GlobalAttention
from NMT.Modules import StackedLSTM
from NMT.Modules import StackedGRU
from .DecoderState import RNNDecoderState


class RNNDecoder(nn.Module):
    def __init__(self, 
                trg_embedding,
                rnn_type,
                embedding_size, 
                hidden_size, 
                num_layers=2, 
                attn_type="general",
                bidirectional_encoder=True,
                dropout=0.0):

        super(RNNDecoder, self).__init__()

        self.trg_embedding = trg_embedding

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = embedding_size + hidden_size
        self.bidirectional_encoder = bidirectional_encoder
        
        self.dropout = nn.Dropout(dropout)
        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self.input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

     
        self.attn = GlobalAttention(hidden_size, 
                                    attn_type=attn_type)

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        if rnn_type == "LSTM":
            stacked_cell = StackedLSTM
        elif rnn_type == "GRU":
            stacked_cell = StackedGRU
        else:
            raise NotImplementedError
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    def init_decoder_state(self, encoder_state):
        assert self.hidden_size == encoder_state[-1].size(-1)
        if isinstance(encoder_state, tuple):
            # LSTM: encoder_state = (hidden, state)
            return RNNDecoderState(encoder_state)
        else:
            # GRU: encoder_state = state
            return RNNDecoderState(encoder_state)

    def forward(self, trg, encoder_outputs, lengths, state):
        """
        Args:
            trg (`LongTensor`): sequences of padded tokens
                                `[L_t x B x D]`.
            encoder_outputs (`FloatTensor`): vectors from the encoder
                 `[L_s x B x H]`.
            
            lengths (`LongTensor`): the padded source lengths
                `[B]`.
            state (`DecoderState`):
                 decoder state object to initialize the decoder
        Returns:
            (`FloatTensor`,:obj:`nmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[trg_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over source words at each target word
                        `[L_t x B x L_s]`.
        """
        # Run the forward pass of the RNN.
        decoder_outputs = []
        attns = []
        
        encoder_outputs = encoder_outputs.transpose(0, 1)
        trg_embed = self.trg_embedding(trg)

        
        # iterate over each target word
        for t, embed in enumerate(trg_embed.split(1, dim=0)):
            output, attn, state = self.forward_step(
                embed, encoder_outputs, lengths, state)
            decoder_output = self.dropout(output)
            decoder_outputs.append(decoder_output)
            attns.append(attn)
        
        
        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)
        attns = torch.stack(attns)

        return decoder_outputs, state, attns



    

    def forward_step(self, trg_embed, encoder_outputs, lengths, state):
        """
        Input feed concatenates hidden state with input at every time step.

        Args:
            trg_embed (LongTensor): each target token
                                `[1 x B x H]`.
            encoder_outputs (`FloatTensor`): vectors from the encoder
                 `[ B x L_s x H]`.
            
            lengths (`LongTensor`): the padded source lengths
                `[B]`.
            state (`DecoderState`):
                 decoder state object to initialize the decoder
        """
        
        input_feed = state.input_feed        # [1 x B x H]
   
        # teacher forcing
        rnn_input = torch.cat([trg_embed, input_feed], -1)
        rnn_state = state.rnn_state

        rnn_input = rnn_input.squeeze(0) 
        # update rnn state and feed to next RNNCell
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)

        attn_output, attn = self.attn(
                rnn_output, encoder_outputs, lengths=lengths)
        
        # update decoder state and input feed feed 
        state = state.update_state(rnn_state, attn_output.unsqueeze(0))

        return attn_output, attn, state