import time
import argparse

from distutils.util import strtobool


#     parser = ArgumentParser()
#     parser.is_training = True if mode == "train" else False
    
# def set_defaults(parser, **kwarg):
#     parser.set_defaults(**kwarg)

# def get_defaults(parser):
#     return parser.parse_args()
    
# @property
# def sections(parser):
#     print(dir(parser))

def make_parser(training=True):
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    add_data_args(parser)
    add_gpu_args(parser)
    add_embed_args(parser)
    add_common_network_args(parser)
    add_rnn_args(parser)
    add_transformer_args(parser)

    if training:
        add_optim_args(parser)
        add_train_args(parser)
    else:
        add_translate_args(parser)
    return parser

def add_default_args(parser):
    group = parser.add_argument_group('Default')

    group.add_argument('--config', type=str, required=True,
                    help="Path to config")

    group.add_argument('--system', type=str,
                    help="which kind of NMT system to use [RNN, Transformer]")

    group.add_argument('-v', '--verbose', action="store_true",
                    help='verbose mode can print more information.')    

    group.add_argument('--save_log', type=str,
                    help="Path to log file")

    group.add_argument('--save_vocab', type=str,
                    help="Path to vocab files")

    group.add_argument('--save_model', default='model',
                    help="""Pre-trained models""")

    group.add_argument('--start_epoch', type=int, default=1,
                    help='Number of training epochs to start')

    group.add_argument('--checkpoint', default=[], nargs='+', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
    
    

def add_gpu_args(parser):
    group = parser.add_argument_group('GPU')

    group.add_argument('--use_gpu', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

    group.add_argument('--use_cpu', default=False, action='store_true',
                    help="Use CPU.")

def add_data_args(parser):
    group = parser.add_argument_group('Data')

    group.add_argument('--src_lang', type=str,
                        help="source language name suffix")
    
    group.add_argument('--trg_lang', type=str,
                        help="target language name suffix")

    group.add_argument('--data_path', type=str,
                    help="path to datasets")

    group.add_argument('--train_dataset', default='train', type=str,
                    help="""The training dataset""")

    group.add_argument('--valid_dataset', default='dev', type=str,
                    help="""The validation dataset""")

    group.add_argument('--test_dataset', default='test', type=str,
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")

    group.add_argument('--max_seq_len', type=int, default=50,
                        help="Maximum sequence length")

    group.add_argument('--shuffle_data', type=lambda x: bool(strtobool(x)), default=True,
                    help="Whether to shuffle data")

    group.add_argument('--mini_batch_sort_order', type=str, default='decreasing',
                    choices=['decreasing', 'increasing', 'none'],
                    help='Order for sorting mini-batches by length')

def add_embed_args(parser):
    group = parser.add_argument_group('Embedding')

    group.add_argument('--min_freq', type=int, default=5,
                        help="Minimal frequency for the prepared data")

    group.add_argument('--src_embed_dim', type=int, default=512,
                    help='Word embedding size for source.')

    group.add_argument('--trg_embed_dim', type=int, default=512,
                    help='Word embedding size for target.')

    group.add_argument('--share_vocab', action='store_true',
                    help="""sharing vocabulary across languages.""") 

    group.add_argument('--share_embedding', action='store_true',
                    help="""sharing embedding across languages.""")

    group.add_argument('--sparse_embeddings', type=lambda x: bool(strtobool(x)), default=False,
                    help='Whether to use sparse embeddings')


    

def add_train_args(parser):
    group = parser.add_argument_group('Train')

    group.add_argument('--max_decrease_steps', type=int, default=10,
                    help='Number of maximal decreased steps for early stopping.') 

    group.add_argument('--epochs', type=int, default=-1,
                    help='Number of training epochs')

    group.add_argument('--train_batch_size', type=int, default=64,
                    help='Maximum batch size for training')

    group.add_argument('--valid_batch_size', type=int, default=32,
                    help='Maximum batch size for validation')    

    group.add_argument('--report_every', type=int, default=100,
                    help='Report statistics after the determinted number of steps.')

def add_translate_args(parser):
    group = parser.add_argument_group('Translate')
     
    

    

    group.add_argument('--output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")

    

    group.add_argument('--load_epoch', type=int, default=-1,
                    help="""Pre-trained models""")

    group.add_argument('--test_batch_size', type=int, default=32,
                    help='Maximum batch size for testing')

    group.add_argument('--beam_size',  type=int, default=5,
                    help='Beam size')

    group.add_argument('--k_best', type=int, default=1,
                    help='Output K-best translations')

    group.add_argument('--max_length', type=int, default=100,
                    help='Maximum prediction length.')

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument('-stepwise_penalty', action='store_true',
                    help="""Apply penalty at every decoding step.
                    Helpful for summary penalty.""")
    group.add_argument('--length_penalty', default='none',
                    choices=['none', 'wu', 'avg'],
                    help="""Length Penalty to use.""")
    group.add_argument('--coverage_penalty', default='none',
                    choices=['none', 'wu', 'summary'],
                    help="""Coverage Penalty to use.""")
    group.add_argument('--alpha', type=float, default=0.,
                    help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add_argument('--beta', type=float, default=-0.,
                    help="""Coverage penalty parameter""")

    group.add_argument('--block_ngram_repeat', type=int, default=0,
                    help='Block repetition of ngrams during decoding.')
    
    group.add_argument('--ignore_when_blocking', nargs='+', type=str,
                    default=[],
                    help="""Ignore these strings when blocking repeats.
                    You want to block sentence delimiters.""")

    group.add_argument('--replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the
                    source token that had highest attention weight. If
                    phrase_table is provided, it will lookup the
                    identified source token and give the corresponding
                    target token. If it is not provided(or the identified
                    source token does not exist in the table) then it
                    will copy the source token""")


    group.add_argument('--plot_attn', action="store_true",
                    help='Plot attention matrix for each pair')

    group.add_argument('--use_beam_search', action="store_true",
                    help='use beam search for decoding.' )

    group.add_argument('--ensemble', default=False, action='store_true',
                    help="Use ensemble-decoding.")
                    
def add_common_network_args(parser):
    group = parser.add_argument_group('Network')

    group.add_argument('--enc_num_layers', type=int, default=2,
                    help='Number of layers in the encoder')

    group.add_argument('--dec_num_layers', type=int, default=2,
                    help='Number of layers in the decoder')
    
    group.add_argument('--attn_type', type=str, default='general',
                    choices=['dot', 'general', 'mlp'],
                    help="""The attention type to use:
                    dotprod or general (Luong) or MLP (Bahdanau)""")

    group.add_argument('--hidden_size', type=int, default=512,
                    help='Number of hidden states')

    group.add_argument('--dropout', type=float, default=0.3,
                    help="Dropout probability; applied in RNN stacks.")


def add_rnn_args(parser):
    group = parser.add_argument_group('RNN')

    group.add_argument('--bidirectional', action='store_true',
                    help="""bidirectional encoding for encoder.""")

    group.add_argument('--rnn_type', type=str,
                    choices=['LSTM', 'GRU'],
                    help="""The gate type to use in the RNNs""")


    group.add_argument('--residual', action='store_true',
                    help="""using residual RNN.""")



def add_transformer_args(parser):
    group = parser.add_argument_group('Transformer')
    
    group.add_argument('--num_heads', type=int, default=8,
                    help='Number of heads in the MultiHeadedAttention')

    group.add_argument('--inner_hidden_size', type=int, default=1024,
                    help='inner hidden size for the MultiHeadedAttention')

    group.add_argument('--latent_size', type=int, default=512,
                    help='latent_size')

def add_optim_args(parser):
    # Optimization options
    group = parser.add_argument_group('Optimizer')


    group.add_argument('--early_stop', action='store_true',
                    help="""early stop.""")

    group.add_argument('--optim', default='Adam',
                    choices=['SGD', 'Adadelta', 'Adam'],
                    help="""Optimization method.""")
    
    group.add_argument('--max_grad_norm', type=float, default=0,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to
                    max_grad_norm""")
    
    group.add_argument('--lr', type=float, default=1e-4,
                    help="""Starting learning rate.
                    Recommended settings: SDG = 1, Adadelta = 1, 
                    Adam = 0.001""")

    group.add_argument('--lr_decay_rate', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")

    group.add_argument('--start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

    group.add_argument('--decay_method', type=str, default="",
                    choices=['noam'], help="Use a custom decay rate.")
    
    group.add_argument('--warmup_steps', type=float, default=4000,
                    help="""warmup_steps for Adam.""")

    group.add_argument('--alpha', type=float, default=0.9,
                    help="""The alpha parameter used by RMSprop.""")

    group.add_argument('--eps', type=float, default=1e-8,
                    help="""The eps parameter used by RMSprop/Adam.""")

    group.add_argument('--rho', type=float, default=0.95,
                    help="""The rho parameter used by RMSprop.""")

    group.add_argument('--weight_decay', type=float, default=0,
                    help="""The weight_decay parameter used by RMSprop.""")

    group.add_argument('--momentum', type=float, default=0,
                    help="""The momentum parameter used by RMSprop[0]/SGD[0.9].""")

    group.add_argument('--adam_beta1', type=float, default=0.9,
                    help="""The beta1 parameter used by Adam.""")

    group.add_argument('--adam_beta2', type=float, default=0.999,
                    help="""The beta2 parameter used by Adam.""")

    group.add_argument('--label_smoothing', type=float, default=0.1,
                    help="""using label smoothing.""")
    
    group.add_argument('--grad_accum_count', type=int, default=4,
                    help="""using label smoothing.""")








