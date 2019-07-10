import re
import os
import numpy as np
import math
import torch
from collections import Counter, defaultdict
from Utils.utils import aeq
from Utils.log import trace
from itertools import chain
import random


PAD_WORD = '<pad>' # 0
UNK_WORD = '<unk>' # 1
BOS_WORD = '<s>'   # 2
EOS_WORD = '</s>'  # 3


class Vocab(object):
    def __init__(self, min_freq=1, specials=[PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]):
        self.specials = specials
        self.counter = Counter()
        self.stoi = {} 
        self.itos = {}
        self.weight = None
        self.min_freq = min_freq
        self.vocab_size = 0
        #config.min_freq

    @property
    def padding_idx(self):
        return self.stoi[PAD_WORD]

    def make_vocab(self, dataset):
        for x in dataset:
            self.counter.update(x)
        
        if self.min_freq > 1:
            self.counter = {w:i for w, i in filter(
                lambda x:x[1] >= self.min_freq, self.counter.items())}
        
        for w in self.specials:
            if w not in self.stoi:
                self.stoi[w] = self.vocab_size
                self.vocab_size += 1

        for w in self.counter.keys():
            if w not in self.stoi:
                self.stoi[w] = self.vocab_size
                self.vocab_size += 1
        
        self.itos = {i:w for w, i in self.stoi.items()}
        
    def load_pretrained_embedding(self, embed_path, embed_dim):
        self.weight = np.zeros((self.vocab_size, int(embed_dim)))
        with open(embed_path, "r", errors="replace") as embed_fin:
            for line in embed_fin:
                cols = line.rstrip("\n").split()
                w = cols[0]
                if w in self.stoi:
                    val = np.array(cols[1:])
                    self.weight[self.stoi[w]] = val
                else:
                    pass
        embed_fin.close()
        for i in range(1, 2):
            self.weight[i] = np.zeros((embed_dim,))
            # self.weights[i] = np.random.random_sample(
            #     (self.config.embed_dim,))
        self.weight = torch.from_numpy(self.weight)

    def __getitem__(self, key):
        return self.weight[key]

    def __len__(self):
        return self.vocab_size

class DataSet(list):
    def __init__(self, src_txt, trg_txt=None, filtered=True, max_length=50):
        super(DataSet, self).__init__()
        self.filtered = filtered
        self.max_length = max_length
        self.read(src_txt, trg_txt)

    def read(self, src_txt, trg_txt):
        with open(src_txt, "r", encoding='utf-8') as fin_src, \
             open(trg_txt, "r", encoding='utf-8') as fin_trg:
             for sid, (line1, line2) in enumerate(zip(fin_src, fin_trg)):
                src, trg = line1.rstrip("\r\n"), line2.rstrip("\r\n")
                src = src.split()
                trg = trg.split()
                if self.filtered:
                    if len(src) <= self.max_length and \
                                    len(trg) <= self.max_length:
                                self.append((sid, (src, trg)))
                else:
                    self.append((sid, (src, trg)))
        fin_src.close()
        fin_trg.close()

    def _numericalize(self, words, stoi):
        return  [1 if x not in stoi else stoi[x] for x in words] 

    @staticmethod
    def _denumericalize(words, itos):
        return ['UNK' if x not in itos else itos[x] for x in words]

    def numericalize(self, src_w2id, trg_w2id):
        for i, (sid, example) in enumerate(self):
            x, y = example
            x = self._numericalize(x, src_w2id)
            y = self._numericalize(y, trg_w2id)
            self[i] = (sid, (x, y))

class DataBatchIterator(object):
    def __init__(self, src_txt, trg_txt, 
            training=True, shuffle=False, share_vocab=False, 
            batch_size=64, max_length=50, vocab=None,
            mini_batch_sort_order='decreasing'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training
        
        # init dataset
        self.examples = DataSet(src_txt, trg_txt,
                        filtered=training,
                        max_length=max_length)

        # init vocab
        self.share_vocab = share_vocab
        self.src_vocab = Vocab()
        if self.share_vocab:
            self.trg_vocab = self.src_vocab
        else:
            self.trg_vocab = Vocab()
        self.init_vocab(vocab)

        self.examples.numericalize(
                src_w2id=self.src_vocab.stoi, 
                trg_w2id=self.trg_vocab.stoi)
        self.num_batches = math.ceil(len(self.examples)/self.batch_size)
        self.mini_batch_sort_order = mini_batch_sort_order

    def init_vocab(self, path):
        if os.path.isfile(path):
            self.load_vocab(path)
        else:
            self.make_vocab()
            self.save_vocab(path)

    def make_vocab(self):
        trace("Building vocabulary ...")
        self.src_vocab.make_vocab(map(lambda x:x[1][0], self.examples))
        self.trg_vocab.make_vocab(map(lambda x:x[1][1], self.examples))
        

    def load_vocab(self, path):
        trace("Loading vocabulary ...")
        if self.share_vocab:
            self.trg_vocab = self.src_vocab = torch.load(path)
        else:
            self.src_vocab, self.trg_vocab = torch.load(path)

    def get_vocab(self): 
        if self.share_vocab:
            return (self.trg_vocab, )
        else:
            return (self.src_vocab, self.trg_vocab)

    def _pad(self, sentence, max_L, w2id, add_bos=False, add_eos=False):
        if add_bos:
            sentence = [w2id[BOS_WORD]] + sentence
        if add_eos:
            sentence =  sentence + [w2id[EOS_WORD]]
        if len(sentence) < max_L:
            sentence = sentence + [w2id[PAD_WORD]] * (max_L-len(sentence))
        return [x for x in sentence]
  

    def pad_seq_pair(self, samples):
        samples = samples if 'none' == self.mini_batch_sort_order else \
            sorted(
                samples,
                key=lambda x: len(x[1][0]),
                reverse=('decreasing' == self.mini_batch_sort_order)
            )
        pairs = [x for x in  map(lambda x:x[1], samples)]
        sid = [x for x in map(lambda x:x[0], samples)]
        
        src_Ls = [len(pair[0])+2 for pair in pairs]
        trg_Ls = [len(pair[1])+2 for pair in pairs]

        max_trg_Ls = max(trg_Ls)
        max_src_Ls = max(src_Ls)
        src = [self._pad(src, max_src_Ls, self.src_vocab.stoi, 
                add_bos=True, add_eos=True) for src, _ in pairs]
        trg = [self._pad(trg, max_trg_Ls, self.trg_vocab.stoi, 
                add_bos=True, add_eos=True) for _, trg in pairs]
        

        batch = Batch()
        batch.src = torch.LongTensor(src).transpose(0, 1).cuda()
        batch.trg = torch.LongTensor(trg).transpose(0, 1).cuda()
        batch.sid = torch.LongTensor(sid).cuda()
        batch.src_Ls = torch.LongTensor(src_Ls).cuda()
        batch.trg_Ls = torch.LongTensor(trg_Ls).cuda()
        return batch

    def save_vocab(self, path):
        if self.share_vocab:
            torch.save(self.trg_vocab, path)
        else:
            torch.save([self.src_vocab, self.trg_vocab], path)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.examples)    
        total_num = len(self.examples)  
        for i in range(self.num_batches): 
            samples = self.examples[i * self.batch_size: \
                        min(total_num, self.batch_size*(i+1))]
            yield self.pad_seq_pair(samples)

    def __repr__(self):
        info = ""
        if self.share_vocab:
            assert self.src_vocab.vocab_size == self.trg_vocab.vocab_size
            info += "Using shared vocab, "
            info += "Vocab: [{0}], ".format(self.src_vocab.vocab_size)
        else:
            info += "Using independent vocab, "
            info += "Source: [{0}], ".format(self.src_vocab.vocab_size)
            info += "Target: [{0}], ".format(self.trg_vocab.vocab_size)
        info += "Dataset: [{0}] ".format(len(self.examples))
        return info

class Batch(object):
    def __init__(self):
        self.src = None
        self.trg = None
        self.src_Ls = None
        self.trg_Ls = None
        
    def __len__(self):
        return self.src_Ls.size(0)
    @property
    def batch_size(self):
        return self.src_Ls.size(0)

