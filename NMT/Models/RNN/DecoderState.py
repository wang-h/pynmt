import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNNDecoderState(object):
    def __init__(self, rnn_state, beam_size=1):
        """
        Args:
            rnn_state: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if isinstance(rnn_state, tuple):         # LSTM
            self.rnn_state = rnn_state
        else:                                    # GRU
            self.rnn_state = (rnn_state, )

        self.beam_size = beam_size
        
        # Init the input feed [1 x B x H].
        self.input_feed = \
            Variable(
                self.rnn_state[0].data.new_zeros(
                    self.rnn_state[0][-1].size(), 
                    requires_grad=False)
            ).unsqueeze(0)    


    def beam_update_state(self, idx, positions):
        
        p = self.input_feed.view(
                self.input_feed.size(0), 
                self.beam_size, 
                self.input_feed.size(1) // self.beam_size,
                self.input_feed.size(2))[:, :, idx]

        p.data.copy_(p.data.index_select(1, positions))

        for h in self.rnn_state:
            p = h.view(h.size(0), 
                    self.beam_size,
                    h.size(1) // self.beam_size,
                    h.size(2))[:, :, idx]
            p.data.copy_(p.data.index_select(1, positions))

    def update_state(self, rnn_state, input_feed):
        state = RNNDecoderState(rnn_state, self.beam_size)
        state.input_feed = input_feed
        return state

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.beam_size = beam_size
        repeat_func = lambda x: Variable(x.data.repeat(1, beam_size, 1))
        self.rnn_state = tuple(repeat_func(h) for h in self.rnn_state)
        self.input_feed = repeat_func(self.input_feed)