import torch
import torch.nn as nn
import numpy as np
import configparser
import os
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))

mixedsignal = config['DMVMCIM']['MixedSignal']

ram_depth = int(config['DMVMPVCIM']['ArraySize'])
DigitRef3 = config['DMVMPVCIM']['DigitRef3']
DigitRef2 = config['DMVMPVCIM']['DigitRef2']
gmin_cancel = config['DMVMPVDevice']['gmincancel']

hardware = config['Quantization']['hardware']
cellprecision       = int(config['DMVMPVCIM']['cellprecision'])
cycleprecision      = int(config['DMVMPVCIM']['cycleprecision'])
resmap  = config['DMVMPVDevice']['resmap']
with_cellvar = config['DMVMPVCIM']['WithCellVar']

adc_mode = config['DMVMPVADC']['mode']   
share_column = int(config['DMVMPVADC']['share'])
DumpData = config['DMVMPVADC']['dumpdata']
if adc_mode == 'Linear':
    RefFile =  config['DMVMPVADC']['linear_file']
elif adc_mode == 'NLinear':
    RefFile =  config['DMVMPVADC']['nlinear_file']

def CIM_MM(input, inputshift, weight, weightshift, adc_map, name=None):
    
    if hardware == "True" and mixedsignal == 'True':  
        from Accuracy.src.Component.DigitConverter import DigitConverter
        digit_converter = DigitConverter(cell_precision=cellprecision, cycle_precision=cycleprecision,isSRAM=True)
        from Accuracy.src.Component.Digit2Cell import Digit2Cell
        digit2cell = Digit2Cell(cell_precision=cellprecision,resmap=resmap,with_cellvar=with_cellvar)
    
        if adc_mode == "Linear":
            from Accuracy.src.Component.ADC.LinearMode import Linear as adc
        if adc_mode == "NLinear":
            from Accuracy.src.Component.ADC.NLinearMode import NLinear as adc
    
    output = None
    testadc = adc(RefFile,share_column,name)
    filter_size = weight.size()
    input_size = input.size()
    
    if DumpData =="True":
        # only dump ps data
        workingdir = os.getcwd()
        outputdir = workingdir \
                    + '/' + config['ADC']['DumpDataPath'] \
                    + '/' + config['Network']['model'] \
                    + '/' + config['Quantization']['mode'] \
                    + '/' + config['Quantization']['weightprecision'] \
                    + '/' + config['Quantization']['inputprecision'] \
                    + '/' + config['Quantization']['weightmapping'] \
                    + '/' + config['Quantization']['inputmapping'] \
                    + '/' + config['Quantization']['weightsignmapping'] \
                    + '/' + config['Quantization']['inputsignmapping'] \
                    + '/' + config['CIM']['cellprecision'] \
                    + '/' + config['CIM']['cycleprecision'] \
                    + '/' + config['CIM']['arraysize']
        PS_dir = outputdir + '/PS_b/'
        partial_sum_name = PS_dir + name + '.npy'
        partial_sum_dummy1_name = PS_dir + name + '_d1.npy'
        partial_sum_dummy2_name = PS_dir + name + '_d2.npy'
        partial_sum_dummy3_name = PS_dir + name + '_d3.npy'
        ps_array = None
        d1_array = None
        d2_array = None
        d3_array = None
        if not os.path.isdir(PS_dir):
            os.makedirs(PS_dir)

    

    skip_ref1 = skip_ref2 = skip_ref3 = skip_gmin_dummy1 = skip_gmin_dummy2 = False
    if inputshift == 0:
        skip_ref1 = True
        skip_ref3 = True
    if weightshift == 0 and gmin_cancel=="False":
        skip_ref2 = True
        skip_ref3 = True

    weightshift_sign = torch.sign(weightshift)
    inputshift_sign = torch.sign(inputshift)

    if gmin_cancel == "False":
         skip_gmin_dummy1 = True
         skip_gmin_dummy2 = True
    else:
        #if weightshift_sign is negative, the shift column itself could be used as dummy
        if weightshift_sign == -1 and DigitRef2 == "False":
            skip_gmin_dummy1 = True
        if  weightshift_sign == -1 and DigitRef3 == "False":
            skip_gmin_dummy2 = True
        #if weight is npsplit, they will cancel their gmin
        if digit_converter.weight_mapping == "Sign" and digit_converter.weight_mapping  == "NPsplit":
            skip_gmin_dummy1 = True
            skip_gmin_dummy2 = True
            
    weightshift_section_bins, weightshift_section_scales = digit_converter.CellDigit(weightshift*weightshift_sign)
    inputshift_section_bins, inputshift_section_scales = digit_converter.VoltageDigit(inputshift*inputshift_sign)
   

    for k in range(int(np.ceil(filter_size[1] / ram_depth))):
                start_channel = k * ram_depth
                end_channel = min((k + 1) * ram_depth, filter_size[1])

                weight_section = weight[start_channel:end_channel, :].clone()
                input_section  =  input[:, start_channel:end_channel].clone()
                
                input_dummy = torch.ones_like(input[0:1, start_channel:end_channel])
                weight_dummy_ones = torch.ones_like(weight[0:1, start_channel:end_channel])
                
                weight_section_bins, weight_section_scales = digit_converter.CellDigit(weight_section)
                input_section_bins, input_section_scales = digit_converter.VoltageDigit(input_section)
                
                for input_b, scale_in, inputshift_b in zip(input_section_bins, input_section_scales, inputshift_section_bins):
                    for l, (weight_b, scale_w, weightshift_b) in enumerate(zip(weight_section_bins, weight_section_scales,weightshift_section_bins)):

                        weight_b = digit2cell.map2G(weight_b) 
                        
                        partial_output_b_sum = torch.mm(input_b, weight_b)

                        if not skip_ref1:
                            #if inputshift_b there is no need to apply it to the array
                            dummy_partial_output_b_sum1 = torch.mm((input_dummy * inputshift_b), weight_b.transpose(0,1))

                        if not skip_ref2:
                            # the ref2 could be used to shift the weight while cancel the gmin of the normal array with the same input.
                            # if one digit is nonzero, all the other zero digit is used to keep consistency of gmin cancel
                            # if all digits are zero, could still be used as gmin cancel which is controlled by configuration
                            if DigitRef2 == "True":
                               dummy_partial_output_b_sum2 = weightshift_b * input_b.sum(axis=1,keepdims=True)
                            else:
                                weight_dummy = weightshift_b * weight_dummy_ones
                                weight_dummy = digit2cell.map2G(weight_dummy)
                                
                                dummy_partial_output_b_sum2 = torch.mm(input_b, weight_dummy)

                        #if not skip_ref3 and inputshift_b!=0:
                        if not skip_ref3:
                            if DigitRef3 == "True":
                                dummy_partial_output_b_sum3 = weightshift_b * inputshift_b * weight_b.shape[1]
                            else:
                                dummy_partial_output_b_sum3 = torch.mm((input_dummy * inputshift_b), weight_dummy.transpose(0, 1))

                        if not skip_gmin_dummy1 or not skip_gmin_dummy2:
                            weight_dummy = 0 * weight_dummy_ones
                            weight_dummy = digit2cell.map2G(weight_dummy)
                            
                            if not skip_gmin_dummy1:
                                dummy_partial_output_b_gmin = torch.mm(input_b, weight_dummy)
                            if not skip_gmin_dummy2:
                                dummy_partial_output_b_gmin2 = torch.mm((input_dummy * inputshift_b), weight_dummy.transpose(0, 1))

                        
                        if DumpData == "True":
                            x = partial_output_b_sum.cpu().data.numpy().flatten()
                            ps = x
                            if ps_array is None:
                                ps_array = ps[0::200]
                            else:
                                ps_array = np.append(ps_array, ps[0::200])
                                
                            if not skip_ref1:
                                x1 = dummy_partial_output_b_sum1.cpu().data.numpy().flatten()
                                d1 = x1
                                if d1_array is None:
                                    d1_array = d1[0::200]
                                else:
                                    d1_array = np.append(d1_array, d1[0::200]) 
                                    
                            if not skip_ref2 and DigitRef2 == "False":
                                x2 = dummy_partial_output_b_sum2.cpu().data.numpy().flatten()
                                d2 = x2
                                if d2_array is None:
                                    d2_array = d2[0::200]
                                else:
                                    d2_array = np.append(d2_array, d2[0::200])
                                    
                            if not skip_ref3 and DigitRef3 == "False":
                                x3 = dummy_partial_output_b_sum3.cpu().data.numpy().flatten()
                                d3 = x3
                                if d3_array is None:
                                    d3_array = d3[0::200]
                                else:
                                    d3_array = np.append(d3_array, d3[0::200])
                        
                        
                        # only for debug
                        partial_output_b_sum = testadc.ADC_compute(partial_output_b_sum)
                        # partial_output_b_sum = partial_output_b_sum  
                        
                        #print("here", partial_output_b_sum[ 0, 0:10])
                        if not skip_ref1:
                            dummy_partial_output_b_sum1 = testadc.ADC_compute(dummy_partial_output_b_sum1) 
                            partial_output_b_sum += dummy_partial_output_b_sum1*inputshift_sign
                        if not skip_ref2:
                            #print("there", dummy_partial_output_b_sum2[ 0, 0:10])
                            if DigitRef2 != "True":
                                dummy_partial_output_b_sum2 = testadc.ADC_compute_ref(dummy_partial_output_b_sum2) 
                            #print("there", dummy_partial_output_b_sum2[0, 0:10])
                            partial_output_b_sum += dummy_partial_output_b_sum2 * weightshift_sign
                        if not skip_ref3:
                            if DigitRef3 != "True":
                                dummy_partial_output_b_sum3 = testadc.ADC_compute_ref(dummy_partial_output_b_sum3) 
                            partial_output_b_sum += dummy_partial_output_b_sum3 * weightshift_sign * inputshift_sign
                        if not skip_gmin_dummy1:
                            dummy_partial_output_b_gmin = testadc.ADC_compute_ref(dummy_partial_output_b_gmin)
                            partial_output_b_sum -= dummy_partial_output_b_gmin
                        if not skip_gmin_dummy2:
                            dummy_partial_output_b_gmin2 = testadc.ADC_compute_ref(dummy_partial_output_b_gmin2) 
                            partial_output_b_sum -= dummy_partial_output_b_gmin2 *inputshift_sign
                        if output is None:
                            output = partial_output_b_sum * scale_w * scale_in
                        else:
                            output = output + partial_output_b_sum * scale_w * scale_in

    if DumpData == "True":
        if not os.path.exists(partial_sum_name):
            np.save(partial_sum_name, ps_array)
            if not skip_ref1:
                np.save(partial_sum_dummy1_name, d1_array)
            if not skip_ref2 and DigitRef2 == "False":
                np.save(partial_sum_dummy2_name, d2_array)
            if not skip_ref3 and DigitRef3 == "False":
                np.save(partial_sum_dummy3_name, d3_array)
                
    return output.float()