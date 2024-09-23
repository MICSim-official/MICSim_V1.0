import configparser
import os
import csv
import numpy as np
import torch

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))

class Digit2Cell():
    def __init__(self, cell_precision, resmap, with_cellvar):
        self.cell_precision = cell_precision
        self.resmap = resmap
        self.with_cellvar = with_cellvar
        self.num_levels = 2 ** self.cell_precision
        self.pure_digtal_level = np.zeros(self.num_levels)
        self.resistance_level  = np.zeros(self.num_levels)
        self.level_variation   = np.zeros(self.num_levels)
        with open(self.resmap, mode='r') as inp:
            reader = csv.reader(inp)
            level_counter = 0
            for rows in reader:
                self.pure_digtal_level[level_counter] = float(rows[0])
                self.resistance_level[level_counter] = float(rows[1])
                self.level_variation[level_counter] = float(rows[2])
                self.Rtx =  float(rows[3])
                level_counter += 1        
        self.Rmax = self.resistance_level[0]
        self.Rmin = self.resistance_level[1]
        self.Gmin = 1/(self.Rmax + self.Rtx)
        self.Gmax = 1/(self.Rmin + self.Rtx)
        self.delta_g = (self.Gmax - self.Gmin)/(self.num_levels-1)
    
    def map2G(self, weight_b):
        weight_g_clean = weight_b * self.delta_g + self.Gmin
        # clean means conductance without cell variation
        # take cell variation into consideration
        if self.with_cellvar == 'True':
            weight_std = torch.zeros_like(weight_g_clean)
            for i_cell_level in range(self.num_levels):
                mask = (weight_b == self.pure_digtal_level[i_cell_level])
                weight_std[mask] = self.level_variation[i_cell_level]
            g_variation = torch.normal(0, weight_std)
            cell_g = weight_g_clean * (1+g_variation)
            weight_b =  cell_g/self.delta_g
        else:
            cell_g = weight_g_clean
            weight_b =  cell_g/self.delta_g
            
        return weight_b            