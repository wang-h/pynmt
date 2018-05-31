import os
import sys
import torch
import random
import math
import numpy as np
from Utils.bleu import compute_bleu
from Utils.rouge import rouge
from Utils.log import trace

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))



def check_save_path(path):
    save_path = os.path.abspath(path)
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def check_file_exist(path):
    
    dirname = os.path.dirname(save_path)
    if not os.path.isfile(dirname):
        os.makedirs(dirname)


def report_bleu(reference_corpus, translation_corpus):
   
    bleu, precions, bp, ratio, trans_length, ref_length =\
        compute_bleu([[x] for x in reference_corpus], translation_corpus)
    trace("BLEU: %.2f [%.2f/%.2f/%.2f/%.2f] Pred_len:%d, Ref_len:%d"%(
        bleu*100, *precions, trans_length, ref_length))


def report_rouge(reference_corpus, translation_corpus):
   
    scores = rouge([" ".join(x) for x in translation_corpus], 
            [" ".join(x) for x in reference_corpus])

     
    trace("ROUGE-1:%.2f, ROUGE-2:%.2f"%(
        scores["rouge_1/f_score"]*100, scores["rouge_2/f_score"]*100))