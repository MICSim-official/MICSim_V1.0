import configparser
import os
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))

QuantizationMode = config['Quantization']['mode']
hardware = config['Quantization']['hardware']
weightsignmapping = config['Quantization']['weightsignmapping']
inputsignmapping = config['Quantization']['inputsignmapping']
weightmapping = config['Quantization']['weightmapping']
inputmapping = config['Quantization']['inputmapping']
adc_mode = config['ADC']['mode']
cellprecision = int(config['CIM']['cellprecision'])



import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from Accuracy.src.utils import make_path
import numpy as np
from Accuracy.src.Layers.CIM_Layer.Linear import Linear_ as CIM_Subarray_LINEAR
Logdir = make_path.makepath_logdir()

printerr = config['Debug']['printLinearerr']

if QuantizationMode == 'IBERT':
    from Accuracy.src.Modules.Transformer.Quantizer.IBERTQuantizer import IBERTQuantizer as Quantizer
    per_channel = True
if QuantizationMode == "Q8BERT":
    from Accuracy.src.Modules.Transformer.Quantizer.Q8BERTQuantizer import Q8BERTQuantizer as Quantizer
    per_channel = False
    
from Accuracy.src.Modules.CNN.Quantizer.Quantizer import Quantizer as QConfig
QConfig = QConfig()


if  weightmapping == "Unsign":
    Wsigned = False
elif weightmapping =="Sign":
    Wsigned = True
else:
    raise ValueError("Unknown weightmapping")

if  inputmapping == "Unsign":
    Isigned = False
elif inputmapping =="Sign":
    Isigned = True
else:
    raise ValueError("Unknown weightmapping")

class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
    
class QLinear(nn.Linear):
    """docstring for QLinear."""

    def __init__(self, in_channels, out_channels,
                 bias=True,
                 quantize_weight=True, quantize_input=True, quantize_error=False,
                 name ='QLinear'):
        super(QLinear, self).__init__(in_channels, out_channels, bias)

        self.name = name
        self.dumpname = None
        self.quantizer = Quantizer()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.register_buffer("bias_integer", torch.zeros_like(self.bias))
             
        self.scale = self.quantizer.weight_init(self.weight, factor=1.0)
        self.quantize_weight = quantize_weight
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.quantize_input = quantize_input
        self.quantize_error = quantize_error
            

    def forward(self, input, prev_act_scaling_factor=None):    
        if self.quantize_input:
            if self.training:
                self.quantizer.update_range(input)
            
            input = self.quantizer.input_clamp(input)    
            
        output = self.CIM_Linear(input, self.weight, self.bias, self.scale, self.name, self.quantizer,prev_act_scaling_factor)
        return output

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={} bias={}, quantize_weight={}, quantize_input={}, quantize_error={}'.format(
            self.in_features, self.out_features, self.bias is not None,self.quantize_weight , self.quantize_input, self.quantize_error
        )


    def CIM_Linear(self, input, weight, bias=None, fixed_scale=1, name='FC', quantizer=None,prev_act_scaling_factor=None):

        weightshift = 0.0
        inputshift = 0.0
        if self.quantize_weight:
            #mapping quantized weight to the code bookw
            weight, weightscale, weightrange, weightshift = quantizer.QuantizeWeight(weight=weight,Wsigned=Wsigned,per_channel=per_channel)

        if self.quantize_input:
            #mapping quantized weight to the code book
            input_int,  inputscale, inputrange, inputshift =   quantizer.QuantizeInput(input=input,inputscale=prev_act_scaling_factor,Isigned=Isigned)
            
        if prev_act_scaling_factor is not None:
            prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)

        if self.bias is not None: 
            if prev_act_scaling_factor is not None:
                bias_scaling_factor = weightscale * prev_act_scaling_factor
            else:
                bias_scaling_factor = weightscale * inputscale
            bias_int = quantizer.QuantizeBias(bias, bias_scaling_factor)
             
        if hardware=='True':
            with torch.no_grad(): 
                inputshift = torch.tensor([inputshift])
                weightshift = torch.tensor([weightshift])
                if torch.cuda.is_available():
                    inputshift = inputshift.cuda()
                    weightshift = weightshift.cuda()
            
                original_shape = input.shape
                if len(original_shape) == 3:
                    N, L, H = input.size()
                    input_int = input_int.view(-1, H)
                # print(self.dumpname)
                output = CIM_Subarray_LINEAR(input_int, inputshift, weight, weightshift, name, self.dumpname)
                if printerr == 'True':
                    output_ref = INTLinear(input_int, inputshift, weight, weightshift, bias=None)
                    error = (output_ref - output).sum()
                    print("Linear err=",error)
                
                output = output + bias_int
                
                if len(original_shape) == 3:
                    output = output.view(N, L, self.out_channels)
    
        else:            
            output = INTLinear(input_int, inputshift, weight, weightshift, bias=None)
            output += bias_int
        
        if quantizer.quantization_mode == 'IBERT':
            return output*bias_scaling_factor, bias_scaling_factor
        else: 
            return output*bias_scaling_factor
        
def INTLinear(input, inputshift, weight, weightshift, bias):
    output = F.linear(input, weight, bias)
    if inputshift != 0:
        dummy_input = torch.ones_like(input)
        output += inputshift * F.linear(dummy_input, weight, bias)
    if weightshift != 0:
        dummy_weight = torch.ones_like(weight)
        output += weightshift * F.linear(input, dummy_weight, bias)
    if inputshift != 0 and weightshift != 0 :
        dummy_input = torch.ones_like(input)
        dummy_weight = torch.ones_like(weight)
        output += weightshift * inputshift * F.linear(dummy_input, dummy_weight, bias)
    return output       