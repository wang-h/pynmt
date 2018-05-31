"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

from Utils.log import trace
from Utils.DataLoader import PAD_WORD
from NMT.Models.RNN import RNNModel
#from NMT.Models.CNN import CNNModel
from NMT.Models.Transformer import TransformerModel
from NMT.CheckPoint import CheckPoint

def make_embeddings(config, *vocab):
    """
    Make an Embeddings instance.
    Args:
        vocab (Vocab): words dictionary.
        config: global configuration settings.
    """
    
    if len(vocab) == 2:
        trace("Making independent embeddings ...")
        src_vocab, trg_vocab = vocab
        padding_idx = src_vocab.stoi[PAD_WORD]
        src_embeddings = nn.Embedding(
                            src_vocab.vocab_size, 
                            config.src_embed_dim,
                            padding_idx=padding_idx, 
                            max_norm=None, 
                            norm_type=2, 
                            scale_grad_by_freq=False, 
                            sparse=False)
        trg_embeddings = nn.Embedding(
                            trg_vocab.vocab_size, 
                            config.src_embed_dim, 
                            padding_idx=padding_idx, 
                            max_norm=None, 
                            norm_type=2, 
                            scale_grad_by_freq=False, 
                            sparse=False)
        if config.hard_encoding:
            self.hard_encoding = IntegerEmbedding(embed_dim, hidden_size, 32, 64)
        
    else:
        
        assert config.trg_embed_dim == config.src_embed_dim
        src_vocab = trg_vocab = vocab[0]
        padding_idx = trg_vocab.padding_idx
        src_embeddings = nn.Embedding(
                            src_vocab.vocab_size, 
                            config.src_embed_dim,
                            padding_idx=padding_idx, 
                            max_norm=None, 
                            norm_type=2, 
                            scale_grad_by_freq=False, 
                            sparse=False)
        if config.share_embedding:
            trace("Making shared embeddings ...")
            trg_embeddings = src_embeddings
        else:
            trace("Making independent embeddings ...")
            trg_embeddings = nn.Embedding(
                                trg_vocab.vocab_size, 
                                config.trg_embed_dim, 
                                padding_idx=padding_idx, 
                                max_norm=None, 
                                norm_type=2, 
                                scale_grad_by_freq=False, 
                                sparse=False)
    return src_vocab, trg_vocab, src_embeddings, trg_embeddings



def model_factory(config, checkpoint, *vocab):
    # Make embedding.

    
    src_vocab, trg_vocab, src_embeddings, trg_embeddings = \
        make_embeddings(config, *vocab)
    
    if config.system == "RNN":
        model = RNNModel( 
            src_embeddings, trg_embeddings, 
            trg_vocab.vocab_size, config)

    elif config.system == "Transformer":
        model = TransformerModel(
            src_embeddings, trg_embeddings, 
            trg_vocab.vocab_size, trg_vocab.padding_idx,    
            config)
    # elif config.system == "CNN":
    #     model = CNNModel(
    #         src_embeddings, trg_embeddings, 
    #         trg_vocab.vocab_size, trg_vocab.padding_idx,    
    #         config)
    if checkpoint:
        trace("Loading model parameters from checkpoint: %s." % str(checkpoint))
        cp = CheckPoint(checkpoint)
        model.load_state_dict(cp.state_dict['model'], strict = False)
    
    if config.training:
        model.train()
    else:
        model.eval()

    if config.use_gpu is not None:
        model.cuda()
    else:
        model.cpu()

    return model
