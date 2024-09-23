import csv
import torch
    
class Linear():
    def __init__(self, ref_file, share_column, name, with_ref=True):
        # super(Linear, self).__init__()
        # self.adc_mode = adc_mode
        self.share_column = share_column
        self.ref_file = ref_file
        self.name = name
        self.with_ref = with_ref
        with open(self.ref_file, mode='r') as inp:
            reader = csv.reader(inp)
            dict_from_csv = {rows[0]: rows[1:] for rows in reader}
        self.adc_min = float(dict_from_csv[self.name][0])
        self.adc_max = float(dict_from_csv[self.name][1])
        self.adc_step = float(dict_from_csv[self.name][2])

        
        if self.with_ref:
            self.adc_ref_min = float(dict_from_csv[self.name][3])
            self.adc_ref_max = float(dict_from_csv[self.name][4])
            self.adc_ref_step = float(dict_from_csv[self.name][5])

            
    def ADC_compute(self, input): 
        output = torch.floor((input + self.adc_step/2) / self.adc_step) * self.adc_step
        output = torch.clamp(output, self.adc_min, self.adc_max) 
        return output
    
    def ADC_compute_ref(self, input):
        output = torch.floor((input + self.adc_ref_step/2) / self.adc_ref_step) * self.adc_ref_step
        output = torch.clamp(output, self.adc_ref_min, self.adc_ref_max) 
        return output 