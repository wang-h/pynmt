Pytorch-based Neural Machine Translation System 
==========
The reason that why I create this project is simple. 
State-of-the-art NMT systems are too much complex and hard to understand by the beginners. 

Personally, I like [Opennmt-py](https://github.com/OpenNMT/OpenNMT-py), and respect their contributions.
However, even I am very familiar with Pytorch, I spend more than one month to understand all details:
- data pre-processing pipelines 
- Encoder-Decoder architecture
- beam-search in NMT
- checkpoint resembling 
- why encoder-decoder model does not need word2vec.

Sometimes, we are only interested in the architecture of model construction, 
neither implementation details nor parameter tuning tricks.

Implemented by Pytorch 0.4, though some modules references to OpenNMT-py, most parts (80%) are written by myself.
I also pasted the scripts for WMT 2018, you can start with them to build your own NMT system and evaluate your systems.


REQUIREMENTS
------------
Python version >= 3.6 (recommended)
Pytorch version >= 0.4 (recommended)

Usage
------------
For training, please use a Moses-style configuration file to specify the path and hyper-parameters.
    
     python train.py --config config/nmt.ini

For translation,

    python translate.py --config config/nmt.ini --checkpoint {pretrained_model.pt} -v

I remember that there is a bug when using beam_search = True, you need to set test_batch_size=1 to make the output correct.

For monotonic decoding (without beam_search), you can use any number for test_batch_size.

## Optimizer
-LSTM:

SGD 1.0 with learning_rate_decay as 0.9 (recommended)


-GRU: 

Adam 1e-4 max_grad_norm = 5 (recommended) 

-Transformer: 

Adam 1e-4, grad_accum_count = 4~5, label_smoothing=0.1 (recommended)

## References

1. Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017

2. Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective approaches to attention-based neural machine translation." arXiv preprint arXiv:1508.04025 (2015).

3. Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).

