import torch
import numpy as np
import itertools as it
from torch.optim import Optimizer


class Lookahead(Optimizer):
    def __init__(self, base_optimizer,alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                                for group in self.param_groups]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group,slow_weights in zip(self.param_groups,self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p,q in zip(group['params'],slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha,p.data - q.data)
                p.data.copy_(q.data)
        return loss


def get_optimizer(name, net, args, get_learn_ratio_func, used_look_head = True):

    base_optimizer = None

    init_learn_ratio = get_learn_ratio_func(0, np.array(args.lr_preset), np.array(args.lr_stage))


    if name == "sgd":

        base_optimizer  = torch.optim.SGD(net.parameters(),lr=init_learn_ratio,momentum = 0.9, weight_decay = args.weight_decay)

    elif name == "adam":

        base_optimizer = torch.optim.Adam(net.parameters(),lr=init_learn_ratio, weight_decay = args.weight_decay)

    else:

        assert "Now Support SGD and Adam"


    if used_look_head is True:

        optimizer = Lookahead(base_optimizer=base_optimizer, k=5, alpha=0.5)
    else:
        optimizer = base_optimizer

    # optimizer = base_optimizer


    return optimizer

