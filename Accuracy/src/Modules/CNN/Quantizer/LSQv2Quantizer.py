from .LSQuantizer import LSQuantizer
import torch
import torch.nn as nn
import math
from torch.autograd import Function
from .LSQuantizer import LSQ

class LSQv2(Function):
    @staticmethod
    def forward(ctx, tensor, scale, g, Qn, Qp, shift):
        ctx.save_for_backward(tensor, scale, shift)
        ctx.other = g, Qn, Qp
        
        Qtensor = torch.div((tensor - shift), scale).round().clamp(Qn, Qp)
        # w_q = w_q * alpha + beta
        return Qtensor, scale, shift

    @staticmethod
    def backward(ctx, grad_tensor, grad_scale, grad_shift):
        tensor, scale, shift = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (tensor - shift) / scale
        smaller = (q_w < Qn).float() 
        bigger = (q_w > Qp).float() 
        between = 1.0 - smaller -bigger 
        grad_scale = ((smaller * Qn + bigger * Qp + 
            between * (q_w.round()) - between * q_w)* grad_tensor * g).sum().unsqueeze(dim=0)
        grad_shift = ((smaller + bigger) * grad_tensor * g).sum().unsqueeze(dim=0)
        grad_tensor = between * grad_tensor
        return grad_tensor, grad_scale,  None, None, None, grad_shift
    
class LSQPlusuantizer(LSQuantizer):
    
    def __init__(self):
        super(LSQPlusuantizer, self).__init__()
        # self.weight_scale  = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.input_scale   = nn.Parameter(torch.ones(1), requires_grad=True)
        self.input_shift = nn.Parameter(torch.tensor([float(-1e-9)]), requires_grad=True)

    def weight_init(self, weight, bits_W=None,factor=2.0, mode="fan_in"):
        scale = 1.0
        return scale    
    
    def update_range(self, input):
        pass
    
    def input_clamp(self, input):
        return input  
                      
    def QuantizeWeight(self, weight, bits=None, Wsigned=True):    
        self.weight_g = 1.0/math.sqrt(weight.numel() * self.WQp)
        mean = torch.mean(weight.detach())
        std = torch.std(weight.detach())
        div = 2**self.weight_precision - 1
            # self.weight_scale.data = max([torch.abs(mean-3*std), torch.abs(mean + 3*std)])/div
        value = (max([torch.abs(mean-3*std), torch.abs(mean + 3*std)])/div)
        self.weight_scale.data = torch.tensor([value]).cuda()
        weight, weightscale = LSQ.apply(weight, self.weight_scale, self.weight_g, self.WQn, self.WQp)
        weightrange = []
        weightshift = 0.0
        
        
        if Wsigned == False:
            weight += 2 ** (self.weight_precision - 1) 
            weightshift  = -(2 ** (self.weight_precision - 1))
        
        return weight, weightscale, weightrange, weightshift 

    def QuantizeInput(self, input, bits=None, Isigned=True):
        
        self.input_g = 1.0/math.sqrt(input.numel() * self.IQp)
        input, inputscale, inputshift = LSQv2.apply(input, self.input_scale, self.input_g, self.IQn, self.IQp, self.input_shift)
     
        inputrange = []   
        
        return input, inputscale, inputrange, inputshift
    
    def QuantizeError(self, error, bits=None, Esigned=True):
        if bits is  None:
            bits= self.error_precision
               
        error, errorscale, errorrange,errorshift = self.Q(error,self.error_precision,signed=Esigned, 
                                                          fixed_range=range,
                                                          odd_stage=True)
        
        return error, errorscale, errorrange, errorshift 

    def quantize_grad(self, x): 
        raise NotImplementedError("use QSGD")
