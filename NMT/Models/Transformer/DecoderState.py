import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TransformerDecoderState(object):
    def __init__(self, src, input=None, previous_inputs=None, beam_size=1):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.input = input
        self.previous_inputs = previous_inputs
        self.beam_size = beam_size

    def update_state(self, input, previous_inputs):
        """
        layer_input : [L_t, B]"
        previous_layer_inputs: [[L_t, B], ...]
        """
        state = TransformerDecoderState(self.src, 
                                        input, 
                                        previous_inputs, 
                                        self.beam_size)
        return state
        
    def beam_update_state(self, idx, positions):
        """
        idx : id in batch
        positions: updateing positions
        """
        batch_size = self.input.size(1) // self.beam_size
      
        p = self.input.view(self.input.size(0), 
                            self.beam_size, 
                            batch_size)[:, :, idx]
        
        p.data.copy_(p.data.index_select(1, positions))

        #print(self.previous_inputs.size())
        pl = self.previous_inputs.view(self.previous_inputs.size(0), 
                            self.beam_size, 
                            batch_size,
                            self.previous_inputs.size(2),
                            self.previous_inputs.size(3))[:, :, idx]

        pl.data.copy_(pl.data.index_select(1, positions))



    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.beam_size = beam_size
        self.src = Variable(
                self.src.data.repeat(1, self.beam_size))
       
