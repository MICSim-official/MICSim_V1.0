import numpy as np
import configparser
import os

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))


weightmapping       = config['Quantization']['weightmapping']
inputmapping        = config['Quantization']['inputmapping']
weightsignmapping   = config['Quantization']['weightsignmapping']
inputsignmapping    = config['Quantization']['inputsignmapping']
weightprecision     = int(config['Quantization']['weightprecision'])
inputprecision      = int(config['Quantization']['inputprecision'])



class DigitConverter():
    def __init__(self, cell_precision, cycle_precision, weight_mapping=weightmapping, input_mapping=inputmapping, 
                 weight_sign_mapping=weightsignmapping, input_sign_mapping=inputsignmapping,
                 weight_precision=weightprecision, input_precision=inputprecision, isSRAM=None):
        
        self.weight_mapping = weightmapping
        self.input_mapping = inputmapping
        self.weight_sign_mapping = weightsignmapping
        self.input_sign_mapping = inputsignmapping
        self.weight_precision = weightprecision
        self.input_precision = inputprecision
        self.cycle_precision = cycle_precision
        self.cell_precision =  cell_precision
        self.isSRAM = isSRAM
    
    def CellDigit(self, weight):
        if self.isSRAM == True:
            if self.cell_precision != 1:
                raise ValueError("SRAM cell precision is 1")
            weight_digit_bins, weight_digit_scales = dec2digit_sign_2s(weight, self.cell_precision, self.weight_precision)
        else:           
            if  (self.weight_mapping) == "Unsign":
                weight_digit_bins, weight_digit_scales = dec2digit_unsign(weight, self.cell_precision, self.weight_precision)
            elif (self.weight_mapping) =="Sign":
                if (self.weight_sign_mapping) == "TwosComp":
                    weight_digit_bins, weight_digit_scales = dec2digit_sign_2s(weight, self.cell_precision, self.weight_precision)
                elif (self.weight_sign_mapping) == "NPsplit":
                    weight_digit_bins, weight_digit_scales = dec2digit_sign_np(weight, self.cell_precision, self.weight_precision)
                else:
                    raise ValueError("Unknown signmapping")
            else:
                raise ValueError("Unknown weightmapping")
        
        return weight_digit_bins, weight_digit_scales
    
    def VoltageDigit(self, input):
        if  (self.input_mapping) == "Unsign":
            input_digit_bins, input_digit_scales = dec2digit_unsign(input, self.cycle_precision, self.input_precision)
        elif (self.input_mapping) =="Sign":
            if (self.input_sign_mapping) == "TwosComp":
                input_digit_bins, input_digit_scales = dec2digit_sign_2s(input, self.cycle_precision, self.input_precision)
            elif (self.input_sign_mapping) == "NPsplit":
                input_digit_bins, input_digit_scales = dec2digit_sign_np(input, self.cycle_precision, self.input_precision)
            else:
                raise ValueError("Unknown signmapping")
        else:
            raise ValueError("Unknown inputmapping") 
        
        return input_digit_bins, input_digit_scales 


def dec2digit_unsign(x, n, N):
    y = x.clone()
    out = []
    scale_list = []
    unit = 2 ** n
    rest = y
    for i in range(int(np.ceil(N / n))):
        y = rest % unit 
        rest = rest // unit 
        out.append(y.clone())
        scale_list.append(unit ** i) # 2^in record the scale infor.
    return out, scale_list

def dec2digit_sign_2s(x, n, N):
    y = x.clone()
    out = []
    scale_list = []
    base = 2 ** (N - 1)

    y[x >= 0] = 0
    y[x < 0] = 1
    rest = x + base * y
    out.append(y.clone())
    scale_list.append(-base)
    unit = 2 ** n
    for i in range(int(np.ceil((N - 1) / n))):
        y = rest % unit
        rest = rest // unit
        out.append(y.clone())
        scale_list.append(unit ** i)
    return out, scale_list

def dec2digit_sign_np(x, n, N):
    
    N = N - 1
    y_p = x.clone()
    y_n = x.clone()
    y_p[y_p <= 0] = 0
    y_n[y_n >= 0] = 0
    y_n = y_n.abs()
    out = []
    scale_list = []
    unit = 2 ** n
    rest_p = y_p
    rest_n = y_n

    for i in range(int(np.ceil(N / n))):
        y_p = rest_p % unit
        y_n = rest_n % unit
        rest_p = rest_p // unit
        rest_n = rest_n // unit
        out.append(y_p.clone())
        scale_list.append(unit ** i)
        out.append(y_n.clone())
        scale_list.append(-unit ** i)
        
    return out, scale_list            