import random
import numpy as np
import neurosim
from Performance.src.Configuration import configuration

class DummyBlock():
    def __init__(self):
        self.area = 0
        self.width = 0
        self.height = 0
        self.readLatency = 0
        self.readDynamicEnergy = 0
        self.numAdderTree = 1
    def CalculateArea(self,*args):
        pass
    def CalculateLatency(self,*args):
        pass
    def CalculatePower(self,*args):
        pass

class Wire(configuration):
    def __init__(self, cell):
        super(Wire,self).__init__() 
        if cell.memCellType == neurosim.MemCellType.SRAM:
            self.wireLengthRow = self.wireWidth * 1e-9 * self.heightInFeatureSizeSRAM
            self.wireLengthCol = self.wireWidth * 1e-9 * self.widthInFeatureSizeSRAM

        else:
            if self.accesstype == 1:
                self.wireLengthRow = self.wireWidth * 1e-9 * self.heightInFeatureSize1T1R
                self.wireLengthCol = self.wireWidth * 1e-9 * self.widthInFeatureSize1T1R
            else :
                self.wireLengthRow = self.wireWidth * 1e-9 *self.heightInFeatureSizeCrossbar
                self.wireLengthCol = self.wireWidth * 1e-9 *self.widthInFeatureSizeCrossbar

        self.Rho *= (1 + 0.00451 * abs(self.temp - 300))
        if self.wireWidth == -1:
            self.unitLengthWireResistance = 1.0 # Use a small number to prevent numerical error for NeuroSim
            self.wireResistanceRow = 0
            self.wireResistanceCol = 0
        else :
            self.unitLengthWireResistance = self.Rho / (self.wireWidth * 1e-9 * self.wireWidth * 1e-9 * self.AR)
            self.wireResistanceRow = self.unitLengthWireResistance * self.wireLengthRow
            self.wireResistanceCol = self.unitLengthWireResistance * self.wireLengthCol
            
def calculate_col_resistance(average_activityRowRead_Subarray,average_conductance,weight_array,average_dummy_conductance,conf,cell,wire,resCellAccess, DigitPerWeight,SubarrayCols):
    resistance = []
    conductance = []
    dummy_resistance = []
    dummy_conductance = []
    
    num_row_matrix = weight_array.shape[0]
    # num_col_matrix = weight_array.shape[1]
    num_col_matrix = min(SubarrayCols, weight_array.shape[1] * DigitPerWeight)
    
    activated_row = np.ceil(average_activityRowRead_Subarray * conf.numRowSubArray)

    input_list = [1] * int(activated_row) + [0] * int(num_row_matrix - activated_row)
    random.seed(250)
    random.shuffle(input_list)

    for j in range(num_col_matrix):
        
        column_g = 0
        total_wire_resistance = 0

        for i in range(num_row_matrix):
            if cell.memCellType == neurosim.MemCellType.RRAM:
                if conf.accesstype == 1: # CMOS access
                    total_wire_resistance = 1.0 / average_conductance + (j + 1) * wire.wireResistanceRow + (num_row_matrix - i) * wire.wireResistanceCol + cell.resistanceAccess
                    # total_wire_resistance = 1.0 / average_conductance
                    # print("temp1:", 1.0 / average_conductance)
                    # print("temp2:",(j + 1) * wire.wireResistanceRow + (num_row_matrix - i) * wire.wireResistanceCol + cell.resistanceAccess)
                else:
                    total_wire_resistance = 1.0 / average_conductance + (j + 1) * wire.wireResistanceRow + (num_row_matrix - i) * wire.wireResistanceCol
            elif cell.memCellType == neurosim.MemCellType.FeFET:
                total_wire_resistance = 1.0 / average_conductance + (j + 1) * wire.wireResistanceRow + (num_row_matrix - i) * wire.wireResistanceCol
            elif cell.memCellType == neurosim.MemCellType.SRAM:
                total_wire_resistance = resCellAccess + wire.wireResistanceCol

            if input_list[i] == 1:
                column_g += 1.0/total_wire_resistance
                # column_g += 10e-5

        conductance.append(column_g)

    # Convert conductance to resistance
    for i in range(num_col_matrix):
        resistance.append(1.0 / conductance[i])
        
    # dummy    
    if average_dummy_conductance == None:
        return resistance, None
    else:
           
        for j in range(conf.numColMuxed):
            column_g = 0
            total_wire_resistance = 0

            for i in range(num_row_matrix):
                if cell.memCellType == neurosim.MemCellType.RRAM:
                    if conf.accesstype == 1: # CMOS access
                        total_wire_resistance = 1.0 / average_dummy_conductance + (j + 1) * wire.wireResistanceRow + (num_row_matrix - i) * wire.wireResistanceCol + cell.resistanceAccess
                        # total_wire_resistance = 1.0 / average_conductance
                        # print("temp1:", 1.0 / average_conductance)
                        # print("temp2:",(j + 1) * wire.wireResistanceRow + (num_row_matrix - i) * wire.wireResistanceCol + cell.resistanceAccess)
                    else:
                        total_wire_resistance = 1.0 / average_dummy_conductance + (j + 1) * wire.wireResistanceRow + (num_row_matrix - i) * wire.wireResistanceCol
                elif cell.memCellType == neurosim.MemCellType.FeFET:
                    total_wire_resistance = 1.0 / average_dummy_conductance + (j + 1) * wire.wireResistanceRow + (num_row_matrix - i) * wire.wireResistanceCol
                elif cell.memCellType == neurosim.MemCellType.SRAM:
                    total_wire_resistance = resCellAccess + wire.wireResistanceCol

                if input_list[i] == 1:
                    column_g += 1.0/total_wire_resistance

            dummy_conductance.append(column_g)

        # Convert conductance to resistance
        for i in range(conf.numColMuxed):
            dummy_resistance.append(1.0 / dummy_conductance[i])
    
        return resistance, dummy_resistance                