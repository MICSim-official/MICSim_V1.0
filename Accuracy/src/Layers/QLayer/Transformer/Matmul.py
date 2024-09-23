import configparser
import os
import torch
import numpy as np
import csv

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
hardware = config['Quantization']['hardware']
mixedsignal = config['DMVMCIM']['MixedSignal']
printerr = config['Debug']['printMatmulerr']
dump_average_value = config['Quantization']['dumpaveragevalue']
dumpaveragevalue_path = config['Quantization']['dumpaveragevalue_path']


def Matmul(tensor1,tensor1_scale,tensor2,tensor2_scale,tensor1shift = 0.0,tensor2shift = 0.0,type=None,dumpname=None):
    
    if not (isinstance(tensor1_scale, float) and isinstance(tensor2_scale, float)):
        tensor1_scale_correct = tensor1_scale.view(-1)
        tensor2_scale_correct = tensor2_scale.view(-1)
    else:
        tensor1_scale_correct = tensor1_scale
        tensor2_scale_correct = tensor2_scale
        
    tensor1shift = torch.tensor([tensor1shift])
    tensor2shift = torch.tensor([tensor2shift])
    if torch.cuda.is_available():
        tensor1shift = tensor1shift.cuda()
        tensor1shift = tensor1shift.cuda() 
    if hardware=='True' and mixedsignal == 'True':
        if type == 'PV':
            from Accuracy.src.Layers.CIM_Layer.CIM_MM_PV import CIM_MM as CIM_Subarray_MM
            cellprecision = int(config['DMVMPVCIM']['cellprecision'])
            cycleprecision = int(config['DMVMPVCIM']['cycleprecision'])
            resmap  = config['DMVMPVDevice']['resmap']
            with_cellvar = config['DMVMPVCIM']['WithCellVar']
        elif type == 'KQ':
            from Accuracy.src.Layers.CIM_Layer.CIM_MM_KQ import CIM_MM as CIM_Subarray_MM
            cellprecision = int(config['DMVMKQCIM']['cellprecision'])
            cycleprecision = int(config['DMVMKQCIM']['cycleprecision'])
            resmap  = config['DMVMKQDevice']['resmap']
            with_cellvar = config['DMVMKQCIM']['WithCellVar']
        else:
            raise NotImplementedError("ONLY SUPPORT TWO TYPES MVM")
                
        batch_size, num_heads, seq_length, attention_head_size = tensor1.size()
        batch_size2, num_heads2, seq_length2, attention_head_size2 = tensor2.size() 
        attention_scores = torch.zeros(batch_size, num_heads, seq_length, attention_head_size2).cuda()
        # print(dumpname)
        if dump_average_value == 'True':
            from Accuracy.src.Component.DigitConverter import DigitConverter
            from Accuracy.src.Component.Digit2Cell import Digit2Cell
            digit_converter = DigitConverter(cell_precision=cellprecision, cycle_precision=cycleprecision,isSRAM=True)
            digit2cell = Digit2Cell(cell_precision=cellprecision,resmap=resmap,with_cellvar=with_cellvar)
            input_bins, input_scales = digit_converter.VoltageDigit(tensor1)
            input_nonzero_ratio=[]
            for input_b in input_bins:
                input_b = np.array(input_b.cpu())
                nonzero_ratio = (np.count_nonzero(input_b)/input_b.size)
                input_nonzero_ratio.append(nonzero_ratio)
            average_input_nonzero_ratio = sum(input_nonzero_ratio)/len(input_nonzero_ratio)
        
            if not os.path.isdir(dumpaveragevalue_path):
                os.makedirs(dumpaveragevalue_path)
            average_dump_file_name = dumpname + '.csv'
            full_file_path = dumpaveragevalue_path + average_dump_file_name
            with open(full_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Average Input Nonzero Ratio', 'Average Conductance Value'])
                writer.writerow([average_input_nonzero_ratio, 'nan']) 

        for batch in range(batch_size):
            for head in range(num_heads):
                tensor1_mm = tensor1[batch, head]  
                tensor2_mm = tensor2[batch, head] 
                score = CIM_Subarray_MM(tensor1_mm,tensor1shift, tensor2_mm, tensor2shift, None, name='dmm') 
                if printerr == 'True':
                    score_ref = torch.mm(tensor1_mm, tensor2_mm)
                    error=(score - score_ref).sum()
                    print("Matmul error=",error)
                    
                score = score * tensor1_scale_correct * tensor2_scale_correct 
                attention_scores[batch, head] = score
        return attention_scores
    else:
        tensor1_fp = tensor1 * tensor1_scale_correct
        tensor2_fp = tensor2 * tensor2_scale_correct
        result = torch.matmul(tensor1_fp, tensor2_fp)
        return result
    