#!/usr/bin/env python
import os
import argparse
import math
import codecs
import torch
from tqdm import tqdm

from itertools import count


from Utils.log import trace
from Utils.config import Config
from Utils.DataLoader import DataBatchIterator
from NMT.ModelFactory import model_factory

#from NMT.Loss import LossBase
from NMT.Trainer import Statistics
from NMT.translate import BatchTranslator
from Utils.plot import plot_attn
from Utils.utils import report_bleu
from Utils.utils import report_rouge
from train import load_dataset


def main():
    config = Config("translate", training=False)   
    if config.verbose: trace(config)
    torch.backends.cudnn.benchmark = True
    
    test_data = load_dataset(config.test_dataset, 
                            config.test_batch_size, 
                            config, prefix="Translate:")

    
    # Build model.
    vocab = test_data.get_vocab()
    pred_file = codecs.open(config.output+".pred.txt", 'w', 'utf-8')
    
    
    model = model_factory(config, config.checkpoint, *vocab)
    translator = BatchTranslator(model, config, test_data.src_vocab, test_data.trg_vocab)
   

    # Statistics
    counter = count(1)
    pred_list = []
    gold_list = []
    for batch in tqdm(iter(test_data), total=test_data.num_batches):
        
        batch_trans = translator.translate(batch)
        
        for trans in batch_trans:
            if config.verbose:
                sent_number = next(counter)
                trace(trans.pprint(sent_number))
            
            if config.plot_attn:
                plot_attn(trans.src, trans.preds[0], trans.attns[0].cpu())

            pred_file.write(" ".join(trans.preds[0]) + "\n")
            pred_list.append(trans.preds[0])
            gold_list.append(trans.gold)  
    report_bleu(gold_list, pred_list)
    report_rouge(gold_list, pred_list)


if __name__ == "__main__":
    main()
