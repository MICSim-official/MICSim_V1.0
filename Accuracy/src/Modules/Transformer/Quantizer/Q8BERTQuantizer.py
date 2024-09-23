import configparser
import os
from Accuracy.src.Modules.CNN.Quantizer.Quantizer import Quantizer
import torch
import torch.nn as nn
from Accuracy.src.utils import make_path
from torch.autograd import Function

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
Logdir = make_path.makepath_logdir()
bias_percision = int(config['Quantization']['biasprecision'])


class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value


class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
                
class Q8BERTQuantizer(Quantizer):
    
    def __init__(self):
        super(Q8BERTQuantizer, self).__init__()
        alpha=0.9999
        self.ema_input = EMA(alpha)
        self.WM = 2**(self.weight_precision-1) -1
        self.IM = 2**(self.input_precision-1) - 1
        self.bias_bits = bias_percision
        self.BM = 2**(self.bias_bits-1) - 1

    def weight_init(self, weight, bits_W=None,factor=2.0, mode="fan_in"):
        scale = 1.0
        return scale 
    
    def update_range(self, input):
        pass
    
    def input_clamp(self, input):
        return input 
                      
    def QuantizeWeight(self, weight, bits=None, Wsigned=True, per_channel=None):
        if bits is  None:
            bits= self.weight_precision
             
        max_val = torch.max(torch.abs(weight)).item()
        scale = self.WM / max_val
        weightscale = 1.0 / scale
        weight = Round.apply(torch.div(weight, weightscale).clamp(-self.WM, self.WM))
        # test:
        # weight_original = weight * scale
        weightrange = []
        weightshift = 0.0
        
        if Wsigned == False:
            weight += 2 ** (bits - 1) - 1
            weightshift  = -(2 ** (bits - 1) - 1)
        
         
        return weight, weightscale, weightrange, weightshift 

    def QuantizeInput(self, input,inputscale=None, bits=None, Isigned=True):
        if bits is  None:
            bits= self.input_precision
        
        self.ema_input.update(torch.max(torch.abs(input)).item())
        scale = self.IM / self.ema_input.value
        
        inputscale  =  1.0 / scale
        input = Round.apply(torch.div(input, inputscale).clamp(-self.IM, self.IM)) 
        
        inputshift = 0.0
        inputrange = []
        
        return input, inputscale, inputrange, inputshift
    
    def QuantizeBias(self, bias, bias_scale):
        Qbias = Round.apply(torch.div(bias, bias_scale).clamp(-self.BM, self.BM)) 
        return Qbias
    
    def QuantizeError(self, error, bits=None, Esigned=True):
        pass

    def quantize_grad(self, x): 
        raise NotImplementedError("use QSGD")

