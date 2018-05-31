import sys
import time
import math
from Utils.log import trace


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        if self.n_words == 0:
            return 0
        return 100 * (float(self.n_correct) / self.n_words)

    def report(self, current_epoch, idx, num_batches, lr):
        """Write out statistics to stdout.

        Args:
           current_epoch (int): current epoch
           idx (int): current batch index
           num__batch (int): total batches
        """
        report = "Epoch {0:d} [{1:d}/{2:d}], Acc: {3:.2f}; PPL: {4:.2f};"
        report += " Loss: {5:.2f}; lr: {6:.6f}   \r"
        return report.format(current_epoch, idx, num_batches, 
                self.accuracy(), self.ppl(), self.loss, lr)

    def report_and_flush(self, current_epoch, idx, num_batches, lr):
        sys.stderr.flush()
        sys.stderr.write(
                self.report(
                    current_epoch, idx, 
                    num_batches, lr))
        sys.stderr.flush()

    def __str__(self):
        string = "Acc: {0:.2f}; PPL: {1:.2f}; Loss: {2:.2f};"
        return string.format(self.accuracy(), self.ppl(), self.loss)

    # def xent(self):
    #     if self.n_words == 0:
    #         return 0
    #     return self.loss / self.n_words

    def ppl(self):
        if self.n_words == 0:
            return 0
        return math.exp(min(float(self.loss) / self.n_words, 100))



