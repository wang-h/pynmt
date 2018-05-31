import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_


from .BaseModel import BaseTransformerModel

class TransformerModel(BaseTransformerModel):
    """
    Core RNN model for NMT.
    {
        Transformer Encoder + Transformer Decoder.
    }
    """
    def __init__(self, src_embedding, trg_embedding,
                 trg_vocab_size, padding_idx, config):
        super(TransformerModel, self).__init__(src_embedding, trg_embedding,
                 trg_vocab_size, padding_idx, config)
        self.context_to_mu = nn.Linear(
                        config.hidden_size, 
                        config.latent_size)
        self.context_to_logvar = nn.Linear(
                        config.hidden_size, 
                        config.latent_size)
        if self.training:
            self.param_init()

    def reparameterize(self, encoder_outputs):
        """
        context [B x 2H]
        """
        #hidden = Variable(encoder_outputs.data, requires_grad=False)
        hidden = encoder_outputs
        mu = self.context_to_mu(hidden)
        logvar = self.context_to_logvar(hidden)
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        return z, mu, logvar

    def translate_step(self, trg, encoder_outputs, lengths, decoder_state):
        output, state, attn = \
            self.decoder(trg, encoder_outputs, lengths, decoder_state)
        return output.squeeze(0), attn.squeeze(0), state

    def forward(self, src, src_lengths, trg, decoder_state=None):
        # encoding side
        encoder_outputs, src = self.encoder(src, src_lengths)

        encoder_outputs, mu, logvar = self.reparameterize(encoder_outputs)
        kld = 0.
        if self.training:
            kld = self.compute_kld(mu, logvar)
        # encoder to decoder
        if decoder_state is None:
            decoder_state = self.decoder.init_decoder_state(src)

        # decoding side
        decoder_outputs, decoder_state, attns = \
            self.decoder(trg, encoder_outputs, src_lengths, decoder_state)
        return decoder_outputs, attns, decoder_state, kld

    def compute_kld(self, mu, logvar):
        kld = -0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        return kld