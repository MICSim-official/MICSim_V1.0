import csv
import torch
import numpy as np


class NLinear():
    def __init__(self, ref_file, share_column,name, with_ref=True):
        # super(NLinear, self).__init__()
        self.share_column = share_column
        self.ref_file = ref_file
        self.name = name
        self.with_ref = with_ref
        dict_from_csv = {}
        with open(self.ref_file, mode='r') as inp:
            reader = csv.reader(inp)
            while True:
                try:
                    layer_name = next(reader)
                    self.centers = next(reader)
                    self.egdes = next(reader)
                    dict_from_csv[layer_name[0]] = [self.centers, self.egdes]
                except:
                    break
        self.centers = torch.tensor(np.array(dict_from_csv[name][0],dtype=float))
        self.egdes  = torch.tensor(np.array(dict_from_csv[name][1],dtype=float))
        if torch.cuda.is_available():
            self.centers = self.centers.cuda().float()
            self.egdes = self.egdes.cuda().float()
    
    def ADC_compute(self, input):
        number_of_level = self.centers.shape[0]
        output = torch.zeros_like(input)
        for i in range(number_of_level):
            if i == 0:
                next_ref = self.egdes[i]
                output = torch.where(input < next_ref, self.centers[i], output)
            elif i == number_of_level - 1:
                current_ref = self.egdes[i-1]
                output = torch.where(input >= current_ref, self.centers[i], output)
            else:
                current_ref = self.egdes[i-1]
                next_ref = self.egdes[i]
                output = torch.where(torch.logical_and(input < next_ref, input >= current_ref), self.centers[i], output )
        
        return output    
        
    def ADC_compute_ref(self, input):
        output = self.ADC_compute(input)
        return output           
        
    