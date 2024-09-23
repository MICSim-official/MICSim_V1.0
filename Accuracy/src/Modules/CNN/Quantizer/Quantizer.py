# read configuration file as global settings
import configparser
import os
import torch
import torch.nn as nn
import math
import numpy as np

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
QuantizationMode    =   config['Quantization']['mode']
WeightPrecision     =   int(config['Quantization']['weightprecision'])
InputPrecision      =   int(config['Quantization']['inputprecision'])
ErrorPrecision      =   int(config['Quantization']['errorprecision'])
GradientPrecision   =   int(config['Quantization']['gradientPrecision'])
WeightMapping       =   config['Quantization']['weightmapping']
InputMapping        =   config['Quantization']['inputmapping']

class Quantizer(nn.Module):
    def __init__(self, quantization_mode=QuantizationMode, weight_precision=WeightPrecision,
                 input_precision=InputPrecision, error_precision=ErrorPrecision,
                 gradient_precision=GradientPrecision,weight_mapping=WeightMapping, input_mapping=InputMapping):
        
        super(Quantizer, self).__init__()
        self.quantization_mode = quantization_mode
        self.weight_precision = weight_precision
        self.input_precision = input_precision
        self.error_precision = error_precision
        self.gradient_precision = gradient_precision 
        if weight_mapping == "Unsign":
            self.Wsigned = False
        elif weight_mapping == "Sign":
            self.Wsigned = True
        else:
            raise ValueError("Unknown weightmapping")
        
        if input_mapping == "Unsign":
            self.Isigned = False
        elif input_mapping == "Sign":
            self.Isigned = True
        else:
            raise ValueError("Unknown inputmapping")
        
        
        
    def Q(self, x, bits, fixed_range=None, forced_shift=None, signed=True, odd_stage=False):
        if bits == -1:
            return x,1,0,0
        if signed and forced_shift is not None:
            raise ValueError("shift is supported for unsigned quantization set only")
        if signed:
            qx,scale,range,shift = self.Q_to_signed_value(x,bits,fixed_range,odd_stage)
        else:
            qx,scale,range,shift = self.Q_to_unsigned_value(x,bits,fixed_range,forced_shift,odd_stage)

        return qx,scale,range,shift
    
    def Q_to_signed_value(self, x, bits, fixed_range, odd_stage):
        if fixed_range is not None:
            min = fixed_range[0]
            max = fixed_range[1]
        else:
            min = x.abs().min()
            max = x.abs().max()
        range = max - min

        try:
            x = torch.clamp(x,-range,range)
        except:
            x = torch.clamp(x,-range.data,range.data)
        if odd_stage:
            num_states = 2. ** bits - 1
            low = -2. ** (bits-1) + 1
            up = 2. ** (bits-1) - 1
        else:
            num_states = 2. ** bits
            low = -2. ** (bits-1)
            up = 2. ** (bits-1) - 1
        scale = (range) / (num_states - 1)
        low =  torch.tensor(low)  * scale
        up =  torch.tensor(up)  * scale
        low = low.cuda()
        up = up.cuda()
        shift = torch.round(torch.tensor(min) / scale) * scale - low
        x = x - shift

        x = torch.clamp(x, low, up)

        x = x / scale

        x = x.round()

        return  x, scale, range, shift/scale
    
    def Q_to_unsigned_value(self, x,bits,fixed_range,forced_shift,odd_stage=False):

        if fixed_range is not None:
            low = fixed_range[0]
            high = fixed_range[1]
        else:
            high = x.max()
            low = x.min()
        #print(fixed_range)
        range = high - low
        #since the quantization should be zero centered, the number of state is 2**N-1 instead of 2N
        if odd_stage:
            num_states = 2. ** bits - 1
        else:
            num_states = 2. ** bits

        scale = (range) / (num_states - 1)
        #print("num_states",num_states)
        try:
            x = torch.clamp(x, low.data, high.data)
        except:
            x = torch.clamp(x, low, high)

        if forced_shift is not None:
            shift =  forced_shift
        else:
            shift = torch.round(torch.tensor(low)/scale) * scale
        x = x - shift
        #print(shift)
        x = torch.clamp(x, 0, range)

        x = x/scale

        x = x.round()
        #print("scale",shift/scale)
        return  x, scale, range, shift/scale    
    
    def S(self, bits):
        return 2. ** (bits - 1)
    
    def C(self, x, bits):
        delta = 1. / self.S(bits)
        upper = 1  - delta
        lower = -1 + delta
        return torch.clamp(x, lower, upper)
    
    def truncated_normal_(self, tensor, mean=0, std=1):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        
    def scale_limit(self, float_std, bits_W):
        delta = 1 / (2.**(bits_W-1))
        if bits_W >2 :
            limit = 1 - delta
            # limit = 0.5
            #limit = 0.75
        else:
            limit = 0.75
        limit_std = limit/math.sqrt(3)
        scale = 2 ** np.round(np.log2(limit_std/float_std))
        return limit,scale    