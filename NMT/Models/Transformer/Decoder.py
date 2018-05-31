import torch
import torch.nn as nn
from NMT.Modules import GlobalAttention
from .Modules import MultiHeadedAttention
from .Modules import PositionwiseFeedForward
from torch.autograd import Variable
from .DecoderState import TransformerDecoderState
import numpy as np
MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    def __init__(self, size, dropout,
                 num_heads=8, hidden_size=1024):

        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
                num_heads, size, dropout=dropout)
        self.context_attn = MultiHeadedAttention(
                num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            size, hidden_size, dropout)
        self.layer_norm_1 = nn.LayerNorm(size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(size, eps=1e-6)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None):

        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None
        query, attn = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask)
        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask)
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn, all_input

    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask

class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".
    """
    def __init__(self, trg_embedding, 
                trg_embed_dim, 
                hidden_size, 
                inner_hidden_size,
                num_layers=4, 
                attn_type="general", 
                dropout=0.0,
                num_heads=8, 
                padding_idx=0):
        super(TransformerDecoder, self).__init__()

        self.trg_embedding = trg_embedding
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        
        self.transformer_layers = \
                nn.ModuleList(
                    [TransformerDecoderLayer(
                        trg_embed_dim, dropout, 
                        num_heads, inner_hidden_size)\
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    
    def init_decoder_state(self, src):
        return TransformerDecoderState(Variable(src))

    def forward(self, trg, encoder_outputs, src_lengths, state):
        """
        Args:
            trg (`LongTensor`): sequences of padded tokens
                                `[L_t x B ]`.
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
        src_words = state.src                               # [L_s, B]             
        src_words = src_words.transpose(0,1).contiguous()
        src_batch, src_len = src_words.size()
        
        trg_words = trg
        trg_words = trg_words.transpose(0,1).contiguous()
        trg_batch, trg_len  = trg_words.size()
        assert (trg_batch == src_batch)

        
        if state.input is not None:
            trg = torch.cat([state.input, trg], 0)
        
        encoder_outputs = encoder_outputs.transpose(0, 1)

        # Run the forward pass of the TransformerDecoder.
        trg_embed = self.trg_embedding(trg)
        if state.input is not None:
            trg_embed = trg_embed[-1:, ]
        # assert trg_embed.dim() == 3  # len x batch x embedding_dim
        
        output = trg_embed.transpose(0, 1).contiguous()
        # B, L_t, H
        src_pad_mask = src_words.data\
                                .eq(self.padding_idx)\
                                .unsqueeze(1) \
                                .expand(src_batch, trg_len, src_len)

        
        trg_pad_mask = trg_words.data\
                                .eq(self.padding_idx)\
                                .unsqueeze(1) \
                                .expand(trg_batch, trg_len, trg_len)

        
        
        saved_inputs = []
        for i in range(self.num_layers):
            input = None
            if state.input is not None:
                input = state.previous_inputs[i]
            output, attn, all_input \
                = self.transformer_layers[i](
                    output, encoder_outputs,
                    src_pad_mask, trg_pad_mask,
                    previous_input=input)
            saved_inputs.append(all_input)
        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)

        # Process the result and update the attentions.
        output = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        # Update the state.
        state = state.update_state(trg, saved_inputs)
        return output, state, attn
