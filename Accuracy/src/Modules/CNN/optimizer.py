import torch
from torch import  nn
from torch.nn.modules import Module
import configparser
import os

# ==================configuration================================
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
optimizer_type = config['Training']['optimizer']
loss_func_type = config['Training']['lossFunc']
lr = float(config['Training']['learning_rate'])
bn_lr = float(config['Training']['bn_learning_rate'])
momentum = float(config['Training']['momentum'])
GradientPrecision = int(config['Quantization']['gradientPrecision'])
# ==================configuration================================

def optimizer(model):
    if optimizer_type == "SGD":
        #opt = torch.optim.SGD(model.parameters(),  lr,
        #                       momentum=momentum, weight_decay=1e-4)
        opt = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=5e-4)
    elif optimizer_type == "ADAM":
        opt = torch.optim.Adam(model.parameters(), lr)

    elif optimizer_type == "QSGD":

        #import self_optimizer.sgd as optimizer
        #opt = optimizer.SGD(model.parameters(), lr=lr,
        #                            momentum=0.9, weight_decay=5e-4)
        import Accuracy.src.self_optimizer.QSGD as optimizer
        opt = optimizer.SGD(model.parameters(), lr=lr,
                            momentum=momentum , wl_grad= GradientPrecision,  weight_decay=5e-4,
                            bn_lr=bn_lr)
    else:
        raise ValueError("Unknown optimizer type")
    return opt

def loss_func():
    if loss_func_type == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_func_type == "SSE":
        criterion = SSE()
    else:
        raise ValueError("Unknown loss_func type")
    return criterion


class SSE(Module):
    def __init__(self):
        super(SSE, self).__init__()
    def forward(self,logits,label):
        target = torch.zeros_like(logits)
        target[torch.arange(target.size(0)).long(), label] = 1
        out = 0.5 * ((logits - target) ** 2).sum()
        return out