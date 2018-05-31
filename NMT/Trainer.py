import sys
import math
import torch
import torch.nn as nn
from itertools import count
import torch.nn.functional as F

from Utils.log import trace
from NMT.Loss import LossBase
from NMT.Loss import LabelSmoothingLoss
from NMT.Optimizer import Optimizer
from NMT.Statistics import Statistics
from NMT.CheckPoint import dump_checkpoint


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model (NMT.Model.NMTModel): NMT model
            config (Config): global configurations
    """

    def __init__(self, model, trg_vocab, padding_idx, config):
        self.model = model

        self.padding_idx = padding_idx
        # self.train_loss = LabelSmoothingLoss(
        #     config, padding_idx, len(trg_vocab),
        #     config.label_smoothing).cuda()
        self.train_loss = LossBase(
            config, padding_idx, len(trg_vocab)).cuda()
        self.valid_loss = LossBase(
            config, padding_idx, len(trg_vocab)).cuda()

        self.optim = Optimizer(config.optim, config)
        self.optim.set_parameters(model.named_parameters())

        self.save_model = config.save_model
        self.last_ppl = float('inf')
        self.steps = 0
        self.max_decrease_steps = config.max_decrease_steps
        self.stop = False
        self.report_every = config.report_every
        self.accum_grad_count = 4
        self.config = config
        self.early_stop = config.early_stop
    def validate(self, valid_data):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        self.model.eval()
        valid_stats = Statistics()
        for batch in iter(valid_data):
            normalization = batch.batch_size
            src, src_lengths = batch.src, batch.src_Ls
            trg, ref = batch.trg[:-1], batch.trg[1:]
            outputs = self.model(src, src_lengths, trg)[0]
            probs = self.model.generator(outputs)
            loss, stats = self.valid_loss.compute(
                probs, ref, normalization)
            valid_stats.update(stats)
            del outputs, probs, stats, loss
        self.model.train()
        return valid_stats

    def train(self, current_epoch, epochs, train_data, valid_data, num_batches):
        """ Train next epoch.
        Args:
            train_data (BatchDataIterator): training dataset iterator
            valid_data (BatchDataIterator): validation dataset iterator
            epoch (int): the epoch number
            num_batches (int): the batch number
        Returns:
            stats (Statistics): epoch loss statistics
        """
        self.model.train()
        
        if self.stop:
            return
        header = '-' * 30 + "Epoch [%d]" + '-' * 30
        trace(header % current_epoch)
        train_stats = Statistics()
        num_batches = train_data.num_batches

        batch_cache = []
        for idx, batch in enumerate(iter(train_data), 1):
            batch_cache.append(batch)
            if len(batch_cache) ==  self.accum_grad_count or idx == num_batches:
                stats = self.train_each_batch(
                        batch_cache, current_epoch, idx, num_batches)
                batch_cache = []
                if idx == train_data.num_batches:
                    train_stats.update(stats)
                if idx % self.report_every == 0 or idx == num_batches:
                    trace(stats.report(current_epoch, idx, num_batches, self.optim.lr))
            if idx % (self.report_every * 10) == 0 and self.early_stop:
                valid_stats = self.validate(valid_data)
                trace("Validation: " + valid_stats.report(current_epoch, idx, num_batches, self.optim.lr))
                if self.early_stop(valid_stats.ppl()):
                    self.stop = True
                    break
        valid_stats = self.validate(valid_data)
        trace(str(valid_stats))
        suffix = ".acc{0:.2f}.ppl{1:.2f}.e{2:d}".format(
            valid_stats.accuracy(),  valid_stats.ppl(), current_epoch)
        self.optim.update_lr(valid_stats.ppl(), current_epoch)
        dump_checkpoint(self.model, self.save_model, suffix)

    def train_each_batch(self, batch_cache, current_epoch, idx, num_batches):
        self.model.zero_grad()
        batch_stats = Statistics()
        normalization = 0
        
        while batch_cache:
            kld = 0
            batch = batch_cache.pop(0)
            src, src_length = batch.src, batch.src_Ls
            trg, ref = batch.trg[:-1], batch.trg[1:]
            normalization += batch.batch_size
            args = self.model(src, src_length, trg)
            outputs= args[0]
            # kld = args[-1]
            probs = self.model.generator(outputs)
            loss, stats = self.train_loss.compute(
                probs, ref, normalization)
            loss.backward(retain_graph=True)
            batch_stats.update(stats)
            del probs, outputs, loss
        self.optim.step()

       
        batch_stats.report_and_flush(
            current_epoch, idx,
            num_batches, self.optim.lr)
        return batch_stats
    def early_stop(self, ppl):
        if ppl < self.last_ppl:
            self.last_ppl = ppl
            self.steps = 0
        else:
            self.steps += 1
        if self.steps >= self.max_decrease_steps:
            return True
        return False
