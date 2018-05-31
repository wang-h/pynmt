import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from Utils.log import trace




class Optimizer(object):
    def __init__(self, method, config):
            
        self.last_ppl = float("inf")
        self.lr = config.lr
        self.original_lr = config.lr
        self.max_grad_norm = config.max_grad_norm
        self.method = method
        self.lr_decay_rate = config.lr_decay_rate
        self.start_decay_at = config.start_decay_at
        self.start_decay = False
        self.alpha = config.alpha
        self._step = 0
        self.decreased_steps = 0
        self.decay_method = config.decay_method
        self.momentum = config.momentum
        self.betas = [config.adam_beta1, config.adam_beta2]
        self.eps = config.eps
        self.warmup_steps = config.warmup_steps
        self.model_size = config.hidden_size
        self.epochs = config.epochs
        
    def set_parameters(self, params):
        self.params = []
        #self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                #if "embed" not in k:
                self.params.append(p)
        if self.method == 'SGD':
            # I recommend SGD when using LSTM.  
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'Adam':
            self.optimizer = optim.Adam(
                                self.params, 
                                lr=self.lr, 
                                betas=self.betas, 
                                eps=1e-9)
        else:
            raise NotImplementedError                       
        # elif self.method == 'Adadelta':
        #     self.optimizer = optim.Adadelta(self.params, lr=self.lr, rho=0.95)
       
        # elif self.method == 'RMSprop':
        #     # does not work properly.
        #     self.optimizer = optim.RMSprop(self.params, lr=self.lr, 
        #         alpha=self.alpha, eps=self.eps, weight_decay=self.lr_decay_rate, 
        #         momentum=self.momentum, centered=False)
    def _set_rate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        self._step += 1
        if self.decay_method == "noam":
            self._set_rate(self.original_lr * (self.model_size ** (-0.5) *
                    min(self._step ** (-0.5), self._step * self.warmup_steps**(-1.5))))
        
        if self.max_grad_norm >0:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def update_lr(self, ppl, epoch):
        if self.start_decay_at is not None and epoch > self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True
        
        if self.start_decay:
            self.lr = self.lr * self.lr_decay_rate
            trace("Decaying learning rate to %g" % self.lr)
        self.last_ppl = ppl
        self.optimizer.param_groups[0]['lr'] = self.lr
        