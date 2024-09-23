import neurosim
import numpy as np
import torch
import csv
from Performance.src.speedup.CM.Tile import Tile
from Performance.src.Modules import DummyBlock
from Performance.src.Configuration import configuration

class Layer():
    def __init__(self,input_param,tech,cell):
        self.Tile = None
        self.AdderTree = neurosim.AdderTree(input_param,tech,cell)
        self.ReLu =  neurosim.BitShifter(input_param,tech,cell) # comment by cong
        self.AdderTreeshift = neurosim.AdderTree(input_param,tech,cell)
        #self.Pool =  m.MaxPooling(input_param,tech,cell)
        self.debug_mode = 1
        self.conf = configuration()
        self.conf.settings_generation()
        self.input_param = self.conf.input_param
        # different layers use different cell
        self.tech = tech
        self.cell = cell

    def Map(self,layer_config):
        # mapping config
        self.batch_size = layer_config[0]
        self.hidden_size=layer_config[1]
        self.attention_head_size=layer_config[2]
        self.sequence_length=layer_config[3]
        self.num_attention_heads=layer_config[4]
        self.averagefile = layer_config[5]
        self.dummy = layer_config[6]
        self.name=layer_config[7]
        self.type=layer_config[8]
        if self.type == 'FC':
            self.cin = self.hidden_size
            self.cout = self.num_attention_heads * self.attention_head_size
            self.inputvector_num = self.sequence_length * self.batch_size
            # Input [self.sequence_length * self.batch_size, self.hidden_size]
            # Weight.T [self.hidden_size, self.num_attention_heads * self.attention_head_size]
            self.InputFeatureMapSize = self.inputvector_num * self.cin
            self.OutputFeatureMapSize = self.sequence_length * self.batch_size * self.num_attention_heads * self.attention_head_size
            self.OP = 2 * self.sequence_length * self.batch_size * self.hidden_size * self.num_attention_heads * self.attention_head_size 
            
        elif self.type == 'MatmulKQ':
            # Q K.T
            # Q(input) = [batch, head, seq_len, head_attention]
            # K(weight) = [batch, head, seq_len, head_attention]
            # result = [batch, head, seq_len, seq_len]
            self.cin = self.attention_head_size
            # this cout is used for mapping
            self.cout = self.sequence_length 
            # for each head and batch, the inputvector number is self.sequence_length 
            self.inputvector_num = self.sequence_length
            # there are num_attention_heads input, each input size is self.inputvector_num * self.cin
            self.InputFeatureMapSize = self.num_attention_heads * self.batch_size * self.attention_head_size *  self.sequence_length 
            # each head submatrix is  self.batch_size * self.attention_head_size * self.sequence_length
            self.OutputFeatureMapSize = self.batch_size * self.num_attention_heads * self.sequence_length * self.sequence_length
            # in'MatmulKQ', weight K.T's size = input Q's size
            self.WeightSize = self.num_attention_heads * self.batch_size * self.attention_head_size *  self.sequence_length
            # for each head,each batch, input[self.sequence_length, self.attention_head_size]
            # weight.T [self.attention_head_size, self.sequence_length]
            self.OP = 2 * self.batch_size * self.num_attention_heads * self.sequence_length * self.attention_head_size * self.sequence_length 

        elif self.type == 'MatmulPV':
            # P V
            # P(input) = [batch, head, seq_len, seq_len]
            # V(weight) = [batch, head, seq_len, head_attention]
            # result = [batch, head, seq_len, head_attention]
            self.cin = self.sequence_length
            # this cout is used for mapping
            self.cout = self.attention_head_size
            # for each head and batch, the inputvector number is self.sequence_length 
            self.inputvector_num = self.sequence_length
            # there are num_attention_heads input, each input size is self.inputvector_num * self.cin
            self.InputFeatureMapSize = self.num_attention_heads * self.batch_size * self.sequence_length *  self.sequence_length 
            # each head submatrix is  self.batch_size * self.sequence_length * self.sequence_length
            self.OutputFeatureMapSize = self.batch_size * self.num_attention_heads * self.sequence_length * self.attention_head_size
            # in'MatmulPV', weight V's size = [seq_len, head_attention]
            self.WeightSize = self.num_attention_heads * self.batch_size * self.attention_head_size *  self.sequence_length
            # for each head,each batch, input[self.sequence_length, self.attention_head_size]
            # weight.T [self.attention_head_size, self.sequence_length]
            self.OP = 2 * self.batch_size * self.num_attention_heads * self.sequence_length * self.sequence_length * self.attention_head_size 
            

        if self.cell.memCellType == neurosim.MemCellType.SRAM:
            self.TileDigitPerWeight = int(np.ceil(self.conf.numBitWeight / self.cell.cellBit))
        else:    
            if  self.conf.weightmapping == "Sign":
                    if self.conf.signmapping == "NPsplit":
                        self.TileDigitPerWeight = int(np.ceil((self.conf.numBitWeight-1)/self.cell.cellBit))
                    else:
                        self.TileDigitPerWeight = int(np.ceil((self.conf.numBitWeight-1)/self.cell.cellBit) + 1)
            else:
                self.TileDigitPerWeight = int(np.ceil(self.conf.numBitWeight / self.cell.cellBit))
                

        
        self.NMmap = False
        
        self.Tile = Tile(self.input_param,self.tech,self.cell)

        if self.type == 'FC':
            self.ArrayPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / self.conf.numColFCSubArray),self.conf.numColFCPE)
            self.ArrayPerFanin =  np.minimum(np.ceil(self.cin / self.conf.numRowFCSubArray),self.conf.numRowFCPE)

            self.DupArrayRow = np.floor(self.conf.numRowFCPE/self.ArrayPerFanin)
            self.DupArrayCol = np.floor(self.conf.numColFCPE/self.ArrayPerFanout)
            
            self.PEPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / (self.conf.numColFCSubArray * self.conf.numColFCPE)),self.conf.numColFCTile)
            self.PEPerFanin =  np.minimum(np.ceil(self.cin  / (self.conf.numRowFCSubArray * self.conf.numRowFCPE)),self.conf.numRowFCTile)

            self.DupPERow = np.floor(self.conf.numRowFCTile/self.PEPerFanin)
            self.DupPECol = np.floor(self.conf.numColFCTile/self.PEPerFanout)

            self.TilePerFanout = np.ceil(self.cout*self.TileDigitPerWeight / (self.conf.numColFCSubArray * self.conf.numColFCPE * self.conf.numColFCTile))
            self.TilePerFanin  = np.ceil(self.cin  / (self.conf.numRowFCSubArray * self.conf.numRowFCPE * self.conf.numRowFCTile))
            
            self.TileFanoutWdith = self.conf.numColFCSubArray*self.conf.numColFCPE*self.conf.numColFCTile
            self.TileFaninWdith  = self.conf.numRowFCSubArray*self.conf.numRowFCPE*self.conf.numRowFCTile
            
            self.numTiles =  self.TilePerFanout * self.TilePerFanin

            self.Dup = self.DupArrayRow*self.DupArrayCol*self.DupPERow*self.DupPECol
            self.totaldigitonchip = self.cout * self.cin *self.Dup *self.TileDigitPerWeight
            self.totalmemcap =  self.numTiles * self.conf.numColFCPE*self.conf.numRowFCPE*self.conf.numRowFCSubArray*self.conf.numColFCSubArray*self.conf.numRowFCTile *self.conf.numColFCTile
            
        elif self.type == 'MatmulKQ':
            self.ArrayPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / self.conf.numColKQSubArray),self.conf.numColKQPE)
            self.ArrayPerFanin =  np.minimum(np.ceil(self.cin / self.conf.numRowKQSubArray),self.conf.numRowKQPE)

            self.DupArrayRow = np.floor(self.conf.numRowKQPE/self.ArrayPerFanin)
            self.DupArrayCol = np.floor(self.conf.numColKQPE/self.ArrayPerFanout)

            self.PEPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / (self.conf.numColKQSubArray * self.conf.numColKQPE)),self.conf.numColKQTile)
            self.PEPerFanin =  np.minimum(np.ceil(self.cin  / (self.conf.numRowKQSubArray * self.conf.numRowKQPE)),self.conf.numRowKQTile)

            self.DupPERow = np.floor(self.conf.numRowKQTile/self.PEPerFanin)
            self.DupPECol = np.floor(self.conf.numColKQTile/self.PEPerFanout)

            self.TilePerFanout = np.ceil(self.cout*self.TileDigitPerWeight / (self.conf.numColKQSubArray * self.conf.numColKQPE * self.conf.numColKQTile))
            self.TilePerFanin  = np.ceil(self.cin  / (self.conf.numRowKQSubArray * self.conf.numRowKQPE * self.conf.numRowKQTile))
            
            self.TileFanoutWdith = self.conf.numColKQSubArray*self.conf.numColKQPE*self.conf.numColKQTile
            self.TileFaninWdith  = self.conf.numRowKQSubArray*self.conf.numRowKQPE*self.conf.numRowKQTile
            
 
            self.numTiles =  self.TilePerFanout * self.TilePerFanin

            self.Dup = self.DupArrayRow*self.DupArrayCol*self.DupPERow*self.DupPECol
            self.totaldigitonchip = self.cout * self.cin * self.Dup * self.TileDigitPerWeight
            # each tile process one sub-matrix mm
            self.numTiles = self.numTiles * self.num_attention_heads
            self.totaldigitonchip = self.totaldigitonchip * self.num_attention_heads
            self.totalmemcap =  self.numTiles * self.conf.numColKQPE*self.conf.numRowKQPE*self.conf.numRowKQSubArray*self.conf.numColKQSubArray*self.conf.numRowKQTile *self.conf.numColKQTile
        
        elif self.type == 'MatmulPV':
            self.ArrayPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / self.conf.numColPVSubArray),self.conf.numColPVPE)
            self.ArrayPerFanin =  np.minimum(np.ceil(self.cin / self.conf.numRowPVSubArray),self.conf.numRowPVPE)

            self.DupArrayRow = np.floor(self.conf.numRowPVPE/self.ArrayPerFanin)
            self.DupArrayCol = np.floor(self.conf.numColPVPE/self.ArrayPerFanout)

            self.PEPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / (self.conf.numColPVSubArray * self.conf.numColPVPE)),self.conf.numColPVTile)
            self.PEPerFanin =  np.minimum(np.ceil(self.cin  / (self.conf.numRowPVSubArray * self.conf.numRowPVPE)),self.conf.numRowPVTile)

            self.DupPERow = np.floor(self.conf.numRowPVTile/self.PEPerFanin)
            self.DupPECol = np.floor(self.conf.numColPVTile/self.PEPerFanout)

            self.TilePerFanout = np.ceil(self.cout*self.TileDigitPerWeight / (self.conf.numColPVSubArray * self.conf.numColPVPE * self.conf.numColPVTile))
            self.TilePerFanin  = np.ceil(self.cin  / (self.conf.numRowPVSubArray * self.conf.numRowPVPE * self.conf.numRowPVTile))
            
            self.TileFanoutWdith = self.conf.numColPVSubArray*self.conf.numColPVPE*self.conf.numColPVTile
            self.TileFaninWdith  = self.conf.numRowPVSubArray*self.conf.numRowPVPE*self.conf.numRowPVTile
            
            self.numTiles =  self.TilePerFanout * self.TilePerFanin

            self.Dup = self.DupArrayRow*self.DupArrayCol*self.DupPERow*self.DupPECol
            self.totaldigitonchip = self.cout * self.cin * self.Dup * self.TileDigitPerWeight

            self.numTiles = self.numTiles * self.num_attention_heads
            self.totaldigitonchip = self.totaldigitonchip * self.num_attention_heads
            self.totalmemcap =  self.numTiles * self.conf.numColPVPE*self.conf.numRowPVPE*self.conf.numRowPVSubArray*self.conf.numColPVSubArray*self.conf.numRowPVTile *self.conf.numColPVTile
                
        self.MemEfficiency = self.totaldigitonchip/self.totalmemcap
        self.resend_rate = 1
        print("Layer: ",self.name," numTiles" ,self.numTiles,"Dup",self.Dup,"MemEfficiency",self.MemEfficiency)
        

    def Configure(self):
        self.outputprecision = 0
        self.outputwidth = 0
        if self.type == 'FC':
            self.TileNumRows = self.conf.numRowFCTile
            self.TileNumCols = self.conf.numColFCTile
            self.PENumRows = self.conf.numRowFCPE
            self.PENumCols = self.conf.numColFCPE
            self.SubarrayRows = self.conf.numRowFCSubArray
            self.SubarrayCols = self.conf.numColFCSubArray
        elif self.type == 'MatmulKQ':
            self.TileNumRows = self.conf.numRowKQTile
            self.TileNumCols = self.conf.numColKQTile
            self.PENumRows = self.conf.numRowKQPE
            self.PENumCols = self.conf.numColKQPE
            self.SubarrayRows = self.conf.numRowKQSubArray
            self.SubarrayCols = self.conf.numColKQSubArray
        elif self.type == 'MatmulPV':
            self.TileNumRows = self.conf.numRowPVTile
            self.TileNumCols = self.conf.numRowPVTile
            self.PENumRows = self.conf.numRowPVPE
            self.PENumCols = self.conf.numColPVPE
            self.SubarrayRows = self.conf.numRowPVSubArray
            self.SubarrayCols = self.conf.numColPVSubArray        
        
        self.Tile.Configure(self.TileNumRows,self.TileNumCols,self.PENumRows,self.PENumCols,self.SubarrayRows,self.SubarrayCols)
        self.outputprecision = self.Tile.outputprecision
        self.outputwidth = self.Tile.outputwidth*self.numTiles

        # add tile for divide fanin
        if self.TilePerFanin >1:
            self.outputwidth = self.outputwidth / self.TilePerFanin
            self.AdderTree.Configure( int(self.TilePerFanin ) , int(self.outputprecision), int(self.outputwidth),self.conf.clkFreq )
            self.outputprecision = self.outputprecision + np.log2(self.TilePerFanin)
        else:
            self.AdderTree = DummyBlock()

        self.AdderTreeshift = DummyBlock()


    def CalculateArea(self):
        self.BufferArea = 0
        self.ICArea = 0
        self.ArrayArea = 0
        self.DigitArea = 0

        self.Tile.CalculateArea()
        self.AdderTree.CalculateArea(0, self.Tile.width*self.TilePerFanout*self.TileDigitPerWeight, neurosim.AreaModify.NONE)

        self.area = self.Tile.area * self.numTiles
        self.ArrayArea = self.Tile.ArrayArea * self.numTiles
        self.BufferArea = self.Tile.BufferArea * self.numTiles
        self.ICArea = self.Tile.ICArea * self.numTiles
        self.DigitArea = self.Tile.DigitArea * self.numTiles

        self.area += self.AdderTree.area
        self.DigitArea += self.AdderTree.area


        self.width = self.Tile.width
        self.height = self.Tile.height * self.numTiles + self.AdderTree.height

        if self.conf.printareaLayer:
            print("================Layer AREA BREAKDOWN=================")
            print("Layer area: ", self.area)
            print("Layer Buffer area: ", self.BufferArea,"( +",0,")")
            print("Layer IC area: ",     self.ICArea,"( +",0,")")


    def CalculatePerformance(self):
        self.SubArrayreadLatency = 0
        self.SubArrayreadDynamicEnergy = 0
        self.BufferreadLatency = 0
        self.BufferreadDynamicEnergy = 0
        self.ICreadLatency = 0
        self.ICreadDynamicEnergy = 0
        self.DigitreadLatency = 0
        self.DigitreadDynamicEnergy = 0
        self.readDynamicEnergy = 0
        self.readLatency = 0
        self.OPoutputprecision = 0
        
        # add for write
        self.writeLatency = 0
        self.writeDynamicEnergy = 0
        self.numBitLoadin_write = 0
        self.SubArraywriteLatency = 0
        self.SubArraywriteDynamicEnergy = 0
        self.BufferwriteLatency = 0 
        self.BufferwriteDynamicEnergy = 0
        self.ICwriteLatency = 0
        self.ICwriteDynamicEnergy = 0
        
        # add for adc
        self.ADCreadLatency = 0
        self.ADCreadDynamicEnergy = 0
        
        with open(self.averagefile, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  
            for row in reader:
                average_activityRowRead = float(row[0])
                average_Condutance = float(row[1])

        if (self.conf.WeDummyCol == True) and (self.dummy != None) and (self.cell.memCellType != neurosim.MemCellType.SRAM):
            with open(self.dummy, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  
                for row in reader:
                    average_dummy_conductance = float(row[0])
        else:
            average_dummy_conductance = None
            
        
        # use it to calculate dims
        # Input [batch, hiddensize, sequence_length]
        if self.type == 'FC':
            inputmatrix = torch.zeros(self.batch_size, self.cin, self.sequence_length)
            weightmartix = torch.zeros(self.cout, self.cin)
        elif self.type == 'MatmulKQ' or self.type == 'MatmulPV':
            inputmatrix = torch.zeros(1, self.cin, self.sequence_length)
            weightmartix = torch.zeros(self.cout, self.cin)
        
        inputshift = torch.tensor([self.conf.inputshift])
        inputshift = inputshift.float()
        
        if self.conf.weightmapping == "Sign" and self.conf.signmapping == "NPsplit" and self.cell.memCellType != neurosim.MemCellType.SRAM:
                weight_section_p = True    
        else:
                weight_section_p = None

        

        weightmartix = weightmartix.view(self.cout,-1)
            
        input_section = inputmatrix[:, 0:self.TileFaninWdith, :]
        weight_section = weightmartix[0:int(self.TileFanoutWdith/self.TileDigitPerWeight), 0:self.TileFaninWdith]
 
        self.Tile.CalculatePerformance(average_activityRowRead, input_section,inputshift,average_Condutance, weight_section,weight_section_p,
                                average_dummy_conductance,[1,1,self.DupArrayRow,self.DupArrayCol]) 
            
        self.readLatency =  self.Tile.readLatency
        self.SubArrayreadLatency =  self.Tile.SubArrayreadLatency
        self.BufferreadLatency = self.Tile.BufferreadLatency
        self.ICreadLatency = self.Tile.ICreadLatency
        self.DigitreadLatency = self.Tile.DigitreadLatency
        # adc
        self.ADCreadLatency = self.Tile.ADCreadLatency
        
        self.readDynamicEnergy += self.numTiles * self.Tile.readDynamicEnergy
        self.SubArrayreadDynamicEnergy +=  self.numTiles * self.Tile.SubArrayreadDynamicEnergy
        self.BufferreadDynamicEnergy   += self.numTiles * self.Tile.BufferreadDynamicEnergy
        self.ICreadDynamicEnergy       += self.numTiles * self.Tile.ICreadDynamicEnergy
        self.DigitreadDynamicEnergy    +=  self.numTiles * self.Tile.DigitreadDynamicEnergy
        # adc
        self.ADCreadDynamicEnergy += self.numTiles * self.Tile.ADCreadDynamicEnergy
        
        # write
        self.writeLatency = self.Tile.writeLatency
        self.SubArraywriteLatency = self.Tile.SubArraywriteLatency
        self.BufferwriteLatency = self.Tile.BufferwriteLatency
        self.ICwriteLatency = self.Tile.ICwriteLatency
        
        self.writeDynamicEnergy += self.numTiles * self.Tile.writeDynamicEnergy
        self.SubArraywriteDynamicEnergy += self.numTiles * self.Tile.SubArraywriteLatency
        self.BufferwriteDynamicEnergy += self.numTiles * self.Tile.BufferwriteDynamicEnergy
        self.ICwriteDynamicEnergy += self.numTiles * self.Tile.ICwriteDynamicEnergy
        
        self.OPoutputprecision = self.Tile.OPoutputprecision

        if self.TilePerFanin > 1:
            self.AdderTree.CalculateLatency(int(np.ceil(self.TilePerFanout*self.cout*self.conf.batchsize*self.height*self.width/self.TileDigitPerWeight/self.AdderTree.numAdderTree)),int(self.TilePerFanin), 0)
            self.AdderTree.CalculatePower(int(np.ceil(self.TilePerFanout*self.cout*self.conf.batchsize*self.height*self.width/self.TileDigitPerWeight/self.AdderTree.numAdderTree)),int(self.TilePerFanin))
            self.OPoutputprecision += np.ceil(np.log2(self.TilePerFanin))

            self.readLatency += self.AdderTree.readLatency
            self.readDynamicEnergy += self.AdderTree.readDynamicEnergy
