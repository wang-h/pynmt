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
from NMT.CheckPoint import CheckPoint
from NMT.Trainer import Statistics
from NMT.translate import BatchTranslator
from Utils.plot import plot_attn
from Utils.utils import report_bleu
from Utils.utils import report_rouge
from train import load_dataset
from NMT.CheckPoint import dump_checkpoint

def main():
    """main function for checkpoint ensemble."""
    config = Config("ensemble", training=True)
    trace(config)
    torch.backends.cudnn.benchmark = True

    train_data = load_dataset(config.train_dataset,
                              config.train_batch_size,
                              config, prefix="Training:")

    # Build model.
    vocab = train_data.get_vocab()
    model = model_factory(config, config.checkpoint, *vocab)
    cp = CheckPoint(config.checkpoint)
    model.load_state_dict(cp.state_dict['model'])
    dump_checkpoint(model, config.save_model, ".ensemble")


if __name__ == "__main__":
    main()
