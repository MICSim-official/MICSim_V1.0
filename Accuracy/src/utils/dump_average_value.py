import os
import configparser
import numpy as np
import torch
import torch.nn as nn
from cimsim.Component.DigitConverter import DigitConverter
from cimsim.Component.Digit2Cell import Digit2Cell


config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
dumpaveragevalue_path = config['Quantization']['dumpaveragevalue_path']

def dump_average_value(input, weight, dumpname):
    output_path = dumpaveragevalue_path
    file_name = dumpname
    
    
    raise NotImplementedError