import torch
import torch.nn as nn
from torch.autograd import Variable

import NMT
import Utils
from NMT.Statistics import Statistics

import torch.nn.functional as F
from Utils.log import trace


class LossBase(nn.Module):
    """
    Standard NMT CrossEntropy/NLL Loss Computation.
    """

    def __init__(self, config, padding_idx, vocab_size):
        super(LossBase, self).__init__()
        self.padding_idx = padding_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=padding_idx, size_average=False)

        self.num_words = 0
        self.num_correct = 0

    def compute(self, probs, golds, normalization):
        """Compute the forward loss and backpropagate.
        Args:
          probs (FloatTensor) : distribution of output model `[(trg_len x batch) x V]`
          golds (LongTensor) : target examples
          normalization   
        Returns:
            :NMT.Statistics: validation loss statistics

        """

        vocab_size = probs.size(-1)
        loss = self.criterion(probs.view(-1, vocab_size), golds.view(-1))
        loss.div(normalization)
        stats = self.create_stats(float(loss), probs, golds)
        return loss, stats

    def create_stats(self, loss, probs, golds):
        """
        Args:
            loss (`FloatTensor`): the loss computed by the loss criterion.
            scores (`FloatTensor`): a score for each possible output
            target (`FloatTensor`): true targets

        Returns:
            `Statistics` : statistics for this batch.
        """

        preds = probs.data.topk(1, dim=-1)[1]
        non_padding = golds.ne(self.padding_idx)
        correct = preds.squeeze(2).eq(golds).masked_select(non_padding)
        num_words = non_padding.long().sum()
        num_correct = correct.long().sum()
        stats = \
            Statistics(float(loss), int(num_words), int(num_correct))
        return stats


class LabelSmoothingLoss(LossBase):
    def __init__(self, config, padding_idx, vocab_size,
                 label_smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__(
            config, padding_idx, vocab_size)

        self.criterion = nn.KLDivLoss(size_average=False)
        one_hot = torch.randn(1, vocab_size).cuda()
        one_hot.fill_(config.label_smoothing / (vocab_size - 2))
        one_hot[0][self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot)
        self.confidence = 1.0 - config.label_smoothing

    def compute(self, probs, golds, normalization, kld=0.):
        vocab_size = probs.size(-1)
        scores = F.log_softmax(probs.view(-1, vocab_size), dim=-1)
        gtruth = golds.view(-1).data
        mask = torch.nonzero(gtruth.eq(self.padding_idx)).long()
        mask = mask.squeeze()
        log_likelihood = torch.gather(scores.data, 1, gtruth.unsqueeze(1))
        tmp = self.one_hot.repeat(gtruth.size(0), 1).cuda()
        tmp.scatter_(1, gtruth.unsqueeze(1), self.confidence)
        if mask.dim() > 0 and mask.size(0) > 0:
            log_likelihood.index_fill_(0, mask, 0)
            tmp.index_fill_(0, mask, 0)
        gtruth = Variable(tmp, requires_grad=False)
        loss = self.criterion(scores, gtruth) 
        #loss += kld
        loss.div(normalization)
        stats = self.create_stats(float(loss), probs, golds)
        return loss, stats
