import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
ratio = 0.1
class Linear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels,
                 bias=True ):

        super(Linear, self).__init__(in_channels, out_channels, bias)



    def forward(self, input):
        bound = self.weight.std()
        weight = torch.clamp(self.weight,-2*bound,2*bound)
        weight_range = 4 * bound.data
        weight_noise = torch.normal(0, weight_range * ratio, size=weight.size())
        if torch.cuda.is_available():
            weight_noise = weight_noise.cuda()
        weight = weight + weight_noise.detach()
        output = F.linear(input, weight, self.bias )

        return output

