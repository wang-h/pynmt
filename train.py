#!/usr/bin/env python

import os
import sys
import glob
import random
import argparse

import torch
import torch.nn as nn

from Utils.log import trace
from Utils.config import Config
from Utils.DataLoader import DataBatchIterator
from Utils.DataLoader import PAD_WORD


from NMT import Trainer
from NMT import Statistics
from NMT import model_factory
from NMT import dump_checkpoint


def main():
    # Load config.
    config = Config("train", training=True)
    trace(config)
    torch.backends.cudnn.benchmark = True

    # Load train dataset.
    train_data = load_dataset(
        config.train_dataset,
        config.train_batch_size,
        config, prefix="Training:")
    
    # Load valid dataset.
    valid_data = load_dataset(
        config.valid_dataset,
        config.valid_batch_size,
        config, prefix="Validation:")

    # Build model.
    vocab = train_data.get_vocab()
    model = model_factory(config, 
                config.checkpoint, *vocab)
    if config.verbose: trace(model)

    # start training
    trg_vocab = train_data.trg_vocab
    padding_idx = trg_vocab.padding_idx
    trainer = Trainer(model, trg_vocab, padding_idx, config)
    start_epoch = 1
    for epoch in range(start_epoch, config.epochs + 1):
        trainer.train(epoch, config.epochs,
                      train_data, valid_data,
                      train_data.num_batches)
    dump_checkpoint(trainer.model, config.save_model)


def load_dataset(dataset, batch_size, config, prefix):
    # Load training/validation dataset.
    train_src = os.path.join(
        config.data_path, dataset + "." + config.src_lang)
    train_trg = os.path.join(
        config.data_path, dataset + "." + config.trg_lang)
    train_data = DataBatchIterator(
        train_src, train_trg,
        share_vocab=config.share_vocab,
        training=config.training,
        shuffle=config.training,
        batch_size=batch_size,
        max_length=config.max_seq_len,
        vocab=config.save_vocab)
    trace(prefix, train_data)
    return train_data


if __name__ == "__main__":
    main()
