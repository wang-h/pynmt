

import os
import sys
import NMT
import Utils
from Utils.log import trace

import torch
import torch.nn as nn
from torch.autograd import Variable

def dump_checkpoint(model, path, suffix=""):
    checkpoint = CheckPoint(model)
    checkpoint.dump(path, suffix)

class CheckPoint(object):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, checkpoint):
        super(CheckPoint, self).__init__()
        self.state_dict = dict()
        if isinstance(checkpoint, list):   
            self.checkpoint_ensemble(checkpoint)
        else:
            self.state_dict['model'] = self.get_state_dict(checkpoint)
    def checkpoint_ensemble(self, checkpoints):
        total = len(checkpoints)
        params = dict()
        for cp in checkpoints:
            sub_params = self.load(cp)
            for k, p in sub_params.items():
                if k not in params:
                    params[k] = 0.
                params[k] += p
        for k in params.keys():
            params[k] /= total
        self.state_dict['model'] = params

    def load(self, path):
        abspath = os.path.abspath(path)
        if os.path.isfile(abspath):
            saved = torch.load(path)
            if "model" in saved:
                params = saved['model']
            else:
                params = saved
            return params
        else:
            trace("#ERROR! checkpoint file does not exist !")
            sys.exit()

    def dump(self, path, suffix=""):
        torch.save(self.state_dict['model'], '%s%s.pt' % (path, suffix))

    def get_state_dict(self, model):
        real_model = (model.module
                      if isinstance(model, nn.DataParallel)
                      else model)
        model_state_dict = real_model.state_dict()
        return dict(model_state_dict.items())