import torch
from torch.optim.optimizer import Optimizer, required
# from Accuracy.src.Layers.Quantizer.Affine_Fixed_point_Quantizer import Q
import numpy as np

from Accuracy.src.Modules.CNN.Quantizer.Quantizer import Quantizer
quantizer = Quantizer()

class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,wl_grad=8,bn_lr=0):
        self.wl_grad = wl_grad
        self.bn_lr = bn_lr
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    #def step(self, closure=None):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for count,p in enumerate(group['params']):
                #print(p.shape)
                #print(p.size())
                #if len(p.shape) == 0:
                #    print(p)
                #    continue

                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                #print(p.shape)
                #p.add_(d_p , alpha=-group['lr'])

                if len(p.shape) ==1:
                    #print(d_p.max())
                    #print("in bn")
                    p.add_(d_p, alpha=-self.bn_lr)
                elif len(p.shape) ==0:
                    continue
                else:
                    #print("======================")
                    #print(d_p.max())
                    if self.wl_grad != -1:

                        delta_weight = - QG(d_p.clone(), group['lr'], self.wl_grad)
                        #print((0-delta_weight).max())
                        d_p = delta_weight
                        p.add_(d_p, alpha=1)
                        #print(p)
                        #np.save('./temp_w_scale/layer' + str(count) + '.npy', p.cpu().numpy().data)
                        #print(self.wl_grad)
                        #print("===============================")

                        #seems want to requantize the gradient weight(copy) to desired precision. mainly works to avoid floating value weight copy
                        range = (1 - 1 / 2 ** (self.wl_grad - 1))
                        qp, pscale, prange, pshift = quantizer.Q(p, self.wl_grad, signed=True, fixed_range=[-range, range])
                        # qp, pscale, prange, pshift = Q(p, self.wl_grad, signed=True,
                        #                                                   fixed_range=1 - 1 / 2 ** (
                        #                                                               self.wl_grad - 1))
                        p.mul_(0).add_(qp*pscale)

                    else:
                        #p.add_(d_p/d_p.abs().max(), alpha=-group['lr'])
                        p.add_(d_p , alpha=-group['lr'])
        #assert 0>1
        return loss

def S(bits):
    return 2.**(bits-1)

def SR(x):
    r = torch.cuda.FloatTensor(*x.size()).uniform_()
    return torch.floor(x+r)

def shift(x):
    #TODO: edge case, when x contains 0
    return 2.**torch.round(torch.log2(x))

def QG(x, lr, bits_G):
    max_entry = x.abs().max()
    assert max_entry != 0, "QG blow"
    x /= shift(max_entry)
    #x /=max_entry
    norm = lr * x
    norm = SR(norm)
    return norm / S(bits_G)
