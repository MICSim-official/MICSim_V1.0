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
# from Accuracy.src.Layers.ADC.Linear_ADC import generate_adc_linear_adcmap,generate_adc_nlinear_adcmap,generate_adc_linear_offset_adcmap,generate_adc_nonlinear_offset_adcmap
Logdir = make_path.makepath_logdir()


if QuantizationMode == "WAGE" :
    from Accuracy.src.Modules.CNN.Quantizer.WAGEQuantizer import WAGEQuantizer as Quantizer
if QuantizationMode == "WAGEV2" :
    from Accuracy.src.Modules.CNN.Quantizer.WAGEV2Quantizer import WAGEV2Quantizer as Quantizer    
if QuantizationMode == "DynamicFixedPoint":
    from Accuracy.src.Modules.CNN.Quantizer.DFQuantizer import DFQuantizer as Quantizer
if QuantizationMode == "LSQ":
    from Accuracy.src.Modules.CNN.Quantizer.LSQuantizer import LSQuantizer as Quantizer
from Accuracy.src.Modules.CNN.Quantizer.Quantizer import Quantizer as QConfig
QConfig = QConfig()


if  weightmapping == "Unsign":
    Wsigned = False
    array_per_weight = int(np.ceil((QConfig.weight_precision) / cellprecision))
    array_extend = 1
elif weightmapping =="Sign":
    Wsigned = True
    if weightsignmapping == 'TwosComp':
        array_per_weight = int(1 + np.ceil((QConfig.weight_precision - 1) / cellprecision))
        array_extend = 1
    elif weightsignmapping == 'NPsplit':
        array_per_weight = int(np.ceil((QConfig.weight_precision - 1) / cellprecision))
        array_extend = 2
    else:
        raise ValueError("unknown signmapping")
else:
    raise ValueError("Unknown weightmapping")

if  inputmapping == "Unsign":
    Isigned = False
elif inputmapping =="Sign":
    Isigned = True
else:
    raise ValueError("Unknown weightmapping")


class QLinear(nn.Linear):
    """docstring for QLinear."""

    def __init__(self, in_channels, out_channels,
                 bias=False,
                 quantize_weight=True, quantize_input=True, quantize_error=True,
                 name ='QLinear'):
        super(QLinear, self).__init__(in_channels, out_channels, bias)

        self.name = name
        self.quantizer = Quantizer() 
        self.scale = self.quantizer.weight_init(self.weight, factor=1.0)
        self.quantize_weight = quantize_weight
        self.quantize_input = quantize_input
        self.quantize_error = quantize_error
            

    def forward(self, input):    
        if self.quantize_input:
            if self.training:
                self.quantizer.update_range(input)
            
            input = self.quantizer.input_clamp(input)    
            
        output = CIM_Linear.apply(input, self.weight, self.bias, self.scale,
                             self.quantize_weight,self.quantize_input,self.quantize_error, self.name, self.quantizer,self.training)
        return output

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={} bias={}, quantize_weight={}, quantize_input={}, quantize_error={}'.format(
            self.in_features, self.out_features, self.bias is not None,self.quantize_weight , self.quantize_input, self.quantize_error
        )


class CIM_Linear(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, fixed_scale=1,
                quantize_weight =True,quantize_input =True,quantize_error =True, name='FC', quantizer=None,training=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.quantizer = quantizer
        ctx.quantize_weight = quantize_weight
        ctx.quantize_input  = quantize_input
        ctx.quantize_error  = quantize_error
        ctx.fixed_scale = fixed_scale
        # ctx.input_range = input_range
        ctx.name = name
        ctx.training = training
        inputscale  = 1
        weightscale = 1
        inputshift  = 0
        weightshift = 0


        if quantize_weight:
            #mapping quantized weight to the code bookw
            weight, weightscale, weightrange, weightshift = quantizer.QuantizeWeight(weight=weight,Wsigned=Wsigned,train=training)

        if quantize_input:
            #mapping quantized weight to the code book
            input,  inputscale, inputrange, inputshift = quantizer.QuantizeInput(input=input,Isigned=Isigned,train=training)


        if quantize_input and quantize_weight and hardware=='True':
            inputshift = torch.tensor([inputshift])
            weightshift = torch.tensor([weightshift])
            if torch.cuda.is_available():
                inputshift = inputshift.cuda()
                weightshift = weightshift.cuda()

            output = CIM_Subarray_LINEAR(input, inputshift, weight, weightshift, name,name)
        
        else:
            output = INTLinear(input, inputshift, weight, weightshift,  bias)

        return output*weightscale*inputscale/fixed_scale
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias= ctx.saved_tensors
        quantizer = ctx.quantizer
        grad_bias = grad_input = grad_weight = None
        inputscale = 1
        inputshift = 0
        weightscale = 1
        weightshift = 0
        grad_outputscale = 1
        grad_outputshift = 0
        dummy_grad_output = torch.zeros_like(grad_output)
        dummy_input = torch.zeros_like(input)
        dummy_weight = torch.zeros_like(weight)

        if ctx.quantize_weight:
            weight, weightscale, weightrange, weightshift = quantizer.QuantizeWeight(weight=weight,Wsigned=Wsigned)
            dummy_weight = torch.ones_like(weight)

        if ctx.quantize_input:
            input,  inputscale, inputrange, inputshift =   quantizer.QuantizeInput(input=input,Isigned=Isigned) 
            dummy_input = torch.ones_like(input)

        if ctx.quantize_error:
            grad_output = torch.clamp(grad_output/grad_output.abs().max(), -1, 1)
            grad_output, grad_outputscale, grad_outputrange, grad_outputshift = quantizer.QuantizeError(error=grad_output,Esigned=Isigned)
            dummy_grad_output = torch.ones_like(grad_output)

        if ctx.needs_input_grad[0]:
            grad_input = torch.mm(grad_output,weight)
            grad_input += torch.mm(dummy_grad_output,weight) * grad_outputshift
            grad_input += torch.mm(grad_output,dummy_weight) * weightshift
            grad_input += torch.mm(dummy_grad_output,dummy_weight) * grad_outputshift * weightshift

        if ctx.needs_input_grad[1]:
            grad_weight =  torch.mm(grad_output.transpose(0,1),input)
            grad_weight += torch.mm(grad_output.transpose(0,1),dummy_input)* inputshift
            grad_weight += torch.mm(dummy_grad_output.transpose(0,1),input) * grad_outputshift
            grad_weight += torch.mm(dummy_grad_output.transpose(0,1),dummy_input) * inputshift * grad_outputshift

        # return grad_input*weightscale*grad_outputscale, grad_weight*inputscale*grad_outputscale, grad_bias, None, None, None, None, None, None, None
        return grad_input*weightscale*grad_outputscale/ctx.fixed_scale, grad_weight*inputscale*grad_outputscale, grad_bias, None, None, None, None, None, None, None
    
def INTLinear(input, inputshift, weight, weightshift, bias):
    output = F.linear(input, weight, bias).detach()
    if inputshift != 0:
        dummy_input = torch.ones_like(input)
        output += inputshift * F.linear(dummy_input, weight, bias).detach()
    if weightshift != 0:
        dummy_weight = torch.ones_like(weight)
        output += weightshift * F.linear(input, dummy_weight, bias).detach()
    if inputshift != 0 and weightshift != 0 :
        dummy_input = torch.ones_like(input)
        dummy_weight = torch.ones_like(weight)
        output += weightshift * inputshift * F.linear(dummy_input, dummy_weight, bias).detach()
    return output   