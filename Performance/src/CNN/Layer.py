import torch
import csv
import numpy as np
import neurosim
from Performance.src.Modules import DummyBlock
from Performance.src.Configuration import configuration

class Layer():
    def __init__(self,input_param,tech,cell):
        self.Tile = None
        self.AdderTree = neurosim.AdderTree(input_param,tech,cell)
        self.ReLu =  neurosim.BitShifter(input_param,tech,cell)
        self.AdderTreeshift = neurosim.AdderTree(input_param,tech,cell)
        self.debug_mode = 1
        self.conf = configuration()
        self.conf.settings_generation()
        self.input_param = self.conf.input_param
        self.tech = tech
        self.cell = cell

    def Map(self,layer_config):
        # mapping config
        self.k1 = layer_config[0]
        self.k2 = layer_config[1]
        self.cin = layer_config[2]
        self.cout = layer_config[3]
        self.H = layer_config[4]
        self.W = layer_config[5]
        self.s1 = layer_config[6]
        self.s2 = layer_config[7]
        self.pad  =  layer_config[8]
        self.averagefile = layer_config[9]
        self.dummy = layer_config[10]
        self.name =  layer_config[11]
        self.type =  layer_config[12]
        
        self.OP = 2*self.k1*self.k2*self.cin*self.cout
        self.OP = self.OP * (np.floor((self.H+2*self.pad-self.k1+1)/self.s1)*np.floor((self.W+2*self.pad-self.k2+1)/self.s2))
        self.OP = self.OP*self.conf.batchsize
        self.InputFeatureMapSize =self.k1*self.k2*self.cin * (np.floor((self.H+2*self.pad-self.k1+1)/self.s1)*np.floor((self.W+2*self.pad-self.k2+1)/self.s2)) * self.conf.batchsize
        self.OutputFeatureMapSize =self.cout * (np.floor((self.H+2*self.pad-self.k1+1)/self.s1)*np.floor((self.W+2*self.pad-self.k2+1)/self.s2)) * self.conf.batchsize

        
        if  self.conf.weightmapping == "Sign":
                if self.conf.signmapping == "NPsplit":
                    self.TileDigitPerWeight = int(np.ceil((self.conf.numBitWeight-1)/self.cell.cellBit))
                else:
                    self.TileDigitPerWeight = int(np.ceil((self.conf.numBitWeight-1)/self.cell.cellBit) + 1)
        else:
            self.TileDigitPerWeight = int(np.ceil(self.conf.numBitWeight / self.cell.cellBit))
        
        if self.k1*self.k2 == self.conf.numRowNMTile *self.conf.numColNMTile and self.k1*self.k2*self.cin >= self.conf.numRowSubArray:
            # novel mapping, same as NeuroSim 
            self.NMmap = True
        
            self.ArrayPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight /  self.conf.numColSubArray),self.conf.numColNMPE)
            self.ArrayPerFanin =  np.minimum(np.ceil(self.cin  /  self.conf.numRowSubArray),self.conf.numRowNMPE)
            self.DupArrayRow = np.floor(self.conf.numRowNMPE/self.ArrayPerFanin)
            self.DupArrayCol = np.floor(self.conf.numColNMPE/self.ArrayPerFanout)

            self.TilePerFanout = np.ceil(self.cout * self.TileDigitPerWeight / (self.conf.numColSubArray*self.conf.numColNMPE))
            self.TilePerFanin  = np.ceil(self.cin  / (self.conf.numRowSubArray*self.conf.numRowNMPE))
            self.TileFanoutWdith = self.conf.numColSubArray*self.conf.numColNMPE
            self.TileFaninWdith  = self.conf.numRowSubArray*self.conf.numRowNMPE
             

            self.numTiles = self.TilePerFanout * self.TilePerFanin
            self.Dup = self.DupArrayRow*self.DupArrayCol
            self.totaldigitonchip = self.cout * self.cin* self.Dup * self.TileDigitPerWeight
            self.totalmemcap =  self.numTiles * self.conf.numColNMPE*self.conf.numRowNMPE*self.conf.numRowSubArray*self.conf.numColSubArray
            self.MemEfficiency = self.totaldigitonchip/self.totalmemcap
            self.resend_rate = 1/ self.k1
            
            if self.MemEfficiency == 1:
                from Performance.src.speedup.NM.Tile import Tile
            else:
                from Performance.src.NM.Tile import Tile
            
            self.Tile = Tile(self.input_param,self.tech,self.cell)
            
            print( "numTiles" ,self.numTiles,"Dup",self.Dup,"MemEfficiency",self.MemEfficiency,"NM")

        else:
            #map layer as cm
            self.NMmap = False
            
            self.ArrayPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / self.conf.numColSubArray),self.conf.numColCMPE)
            self.ArrayPerFanin =  np.minimum(np.ceil(self.cin * self.k1 * self.k2 / self.conf.numRowSubArray),self.conf.numRowCMPE)
            self.DupArrayRow = np.floor(self.conf.numRowCMPE/self.ArrayPerFanin)
            self.DupArrayCol = np.floor(self.conf.numColCMPE/self.ArrayPerFanout)

            self.PEPerFanout = np.minimum(np.ceil(self.cout * self.TileDigitPerWeight / (self.conf.numColSubArray * self.conf.numColCMPE)),self.conf.numColCMTile)
            self.PEPerFanin =  np.minimum(np.ceil(self.cin * self.k1 * self.k2 / (self.conf.numRowSubArray * self.conf.numRowCMPE)),self.conf.numRowCMTile)
 
            self.DupPERow = np.floor(self.conf.numRowCMTile/self.PEPerFanout)
            self.DupPECol = np.floor(self.conf.numColCMTile/self.PEPerFanin)

            self.TilePerFanout = np.ceil(self.cout*self.TileDigitPerWeight / (self.conf.numColSubArray * self.conf.numColCMPE * self.conf.numColCMTile))
            self.TilePerFanin  = np.ceil(self.cin * self.k1 * self.k2 / (self.conf.numRowSubArray * self.conf.numRowCMPE * self.conf.numRowCMTile))
            
            self.TileFanoutWdith = self.conf.numColSubArray*self.conf.numColCMPE*self.conf.numColCMTile
            self.TileFaninWdith  = self.conf.numRowSubArray*self.conf.numRowCMPE*self.conf.numRowCMTile

            self.numTiles =  self.TilePerFanout * self.TilePerFanin
            self.Dup = self.DupArrayRow*self.DupArrayCol*self.DupPERow*self.DupPECol
            self.totaldigitonchip = self.cout * self.cin* self.k1 * self.k2 *self.Dup *self.TileDigitPerWeight
            self.totalmemcap =  self.numTiles * self.conf.numColCMPE*self.conf.numRowCMPE*self.conf.numRowSubArray*self.conf.numColSubArray*self.conf.numRowCMTile *self.conf.numColCMTile
            self.MemEfficiency = self.totaldigitonchip/self.totalmemcap
            self.resend_rate = 1
            
            if self.MemEfficiency == 1:
                from Performance.src.speedup.CM.Tile import Tile
            else:
                from Performance.src.CM.Tile import Tile
            
            self.Tile = Tile(self.input_param,self.tech,self.cell)
            
            
            print("Layer: ",self.name," numTiles" ,self.numTiles,"Dup",self.Dup,"MemEfficiency",self.MemEfficiency,"CM")

    def Configure(self):
        self.outputprecision = 0
        self.outputwidth = 0
        if self.NMmap == False:
            self.TileNumRows = self.conf.numRowCMTile
            self.TileNumCols = self.conf.numColCMTile
            self.PENumRows = self.conf.numRowCMPE
            self.PENumCols = self.conf.numColCMPE
            self.SubarrayRows = self.conf.numRowSubArray
            self.SubarrayCols = self.conf.numColSubArray
            self.Tile.Configure(self.TileNumRows,self.TileNumCols,self.PENumRows,self.PENumCols,self.SubarrayRows,self.SubarrayCols)
        else:
            self.Tile.Configure()
        
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
        self.height = self.Tile.height * self.numTiles + self.AdderTree.height #+ self.ReLu.height

        if self.conf.printareaLayer:
            print("================PE LEVEL AREA BREAKDOWN=================")
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
        
        # add for adc
        self.ADCreadLatency = 0
        self.ADCreadDynamicEnergy = 0
        
        # read average files
        with open(self.averagefile, mode='r') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                average_activityRowRead = float(row[0])
                average_Condutance = float(row[1])
        
        if self.conf.WeDummyCol == True and self.dummy != None:
            with open(self.dummy, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  
                for row in reader:
                    average_dummy_conductance = float(row[0])
        else:
            average_dummy_conductance = None
            
                    
        # use it to calculate dims
        inputmatrix = torch.zeros(self.conf.batchsize, self.cin, self.H, self.W)
        weightmartix = torch.zeros(self.cout, self.cin, self.k1, self.k2)
        
        inputshift = torch.tensor([self.conf.inputshift])
        inputshift = inputshift.float()

        # the unfold function will unfold the input to the shape [B,k1*k2*cin,numvector], the vector is calcualted from H,W,stride and pad
        unfoldmap = torch.nn.Unfold((self.k1,self.k2), dilation=1, padding=self.pad, stride=self.s1)
        
        if self.conf.weightmapping == "Sign" and self.conf.signmapping == "NPsplit":
            weight_section_p = True    
        else:
            weight_section_p = None

        if self.NMmap:
            if self.MemEfficiency == 1:
                input_section = inputmatrix[:, 0:self.TileFaninWdith,:, :]
                weight_section = weightmartix[0:int(self.TileFanoutWdith/self.TileDigitPerWeight), 0:self.TileFaninWdith,:,:]
                
                input_section =unfoldmap(torch.tensor(input_section)) 
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
            else:
                for fanout_index in range(int(self.TilePerFanout)):
                    for fanin_index in range(int(self.TilePerFanin)):
                        #cut weight matrix according to tiles
                        cinstart = fanin_index   * self.TileFaninWdith
                        cinend = (fanin_index+1) * self.TileFaninWdith
                        coutstart = fanout_index * int(self.TileFanoutWdith/self.TileDigitPerWeight)
                        coutend =(fanout_index+1)* int(self.TileFanoutWdith/self.TileDigitPerWeight)

                        input_section  = inputmatrix[:,cinstart:cinend,:,:]
                        weight_section = weightmartix[coutstart:coutend,cinstart:cinend,:,:]

                        input_section =unfoldmap(torch.tensor(input_section)) 
                        self.Tile.CalculatePerformance(average_activityRowRead, input_section,inputshift,average_Condutance, weight_section,weight_section_p,
                                    average_dummy_conductance,[1,1,self.DupArrayRow,self.DupArrayCol]) 
            
                        if self.Tile.readLatency > self.readLatency:
                            self.readLatency =  self.Tile.readLatency
                            self.SubArrayreadLatency =  self.Tile.SubArrayreadLatency
                            self.BufferreadLatency = self.Tile.BufferreadLatency
                            self.ICreadLatency = self.Tile.ICreadLatency
                            self.DigitreadLatency = self.Tile.DigitreadLatency
                            # adc
                            self.ADCreadLatency = self.Tile.ADCreadLatency
            
                        self.readDynamicEnergy += self.Tile.readDynamicEnergy
                        self.SubArrayreadDynamicEnergy += self.Tile.SubArrayreadDynamicEnergy
                        self.BufferreadDynamicEnergy   += self.Tile.BufferreadDynamicEnergy
                        self.ICreadDynamicEnergy       += self.Tile.ICreadDynamicEnergy
                        self.DigitreadDynamicEnergy    += self.Tile.DigitreadDynamicEnergy
                        # adc
                        self.ADCreadDynamicEnergy += self.Tile.ADCreadDynamicEnergy
            
        else:
            if self.MemEfficiency == 1:
                weightmartix = weightmartix.view(self.cout,-1)
                inputmatrix = unfoldmap(inputmatrix)
                
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
            else:
                #unlike the nm mapping which mainly cut cin and cout with k1,k2 mapped to different PE, the CM view the weight as a 2d array with k1*k2*Cin X Cout

                weightmartix = weightmartix.view(self.cout,-1)
                # The CM unfold input and then cut  according to tile fanin
                # the unfold function will unfold the input to the shape [Batch,k1*k2*cin,numvector], the vector is calcualted from H,W,stride and pad 
                inputmatrix = unfoldmap(inputmatrix)

                for fanout_index in range(int(self.TilePerFanout)):
                    for fanin_index in range(int(self.TilePerFanin)):
                        # cut weight matrix according to tiles
                        cinstart = fanin_index   * self.TileFaninWdith
                        cinend = (fanin_index+1) * self.TileFaninWdith
                        coutstart = fanout_index * int(self.TileFanoutWdith/self.TileDigitPerWeight)
                        coutend =(fanout_index+1)* int(self.TileFanoutWdith/self.TileDigitPerWeight)

                        input_section  = inputmatrix[:,cinstart:cinend,:]
                        weight_section = weightmartix[coutstart:coutend,cinstart:cinend]
                        self.Tile.CalculatePerformance(average_activityRowRead, input_section,inputshift,average_Condutance, weight_section,weight_section_p,
                                                        average_dummy_conductance,[1,1,self.DupArrayRow,self.DupArrayCol]) 
                
                        if self.Tile.readLatency > self.readLatency:
                            self.readLatency = self.Tile.readLatency
                            self.SubArrayreadLatency = self.Tile.SubArrayreadLatency
                            self.BufferreadLatency = self.Tile.BufferreadLatency
                            self.ICreadLatency = self.Tile.ICreadLatency
                            self.DigitreadLatency = self.Tile.DigitreadLatency
                            # adc
                            self.ADCreadLatency = self.Tile.ADCreadLatency
                        self.readDynamicEnergy += self.Tile.readDynamicEnergy
                        self.SubArrayreadDynamicEnergy += self.Tile.SubArrayreadDynamicEnergy
                        self.BufferreadDynamicEnergy += self.Tile.BufferreadDynamicEnergy
                        self.ICreadDynamicEnergy += self.Tile.ICreadDynamicEnergy
                        self.DigitreadDynamicEnergy += self.Tile.DigitreadDynamicEnergy
                        # adc
                        self.ADCreadDynamicEnergy += self.Tile.ADCreadDynamicEnergy

                
        self.OPoutputprecision = self.Tile.OPoutputprecision

        if self.numTiles > 1:

            self.AdderTree.CalculateLatency(int(np.ceil(self.TilePerFanout*self.cout*self.conf.batchsize*self.height*self.width/self.TileDigitPerWeight/(self.s1*self.s2)/self.AdderTree.numAdderTree)),int(self.TilePerFanin), 0)
            self.AdderTree.CalculatePower(int(np.ceil(self.TilePerFanout*self.cout*self.conf.batchsize*self.height*self.width//self.TileDigitPerWeight/(self.s1*self.s2)/self.AdderTree.numAdderTree)),int(self.TilePerFanin))
            self.OPoutputprecision += np.ceil(np.log2(self.TilePerFanin))

            self.readLatency += self.AdderTree.readLatency
            self.readDynamicEnergy += self.AdderTree.readDynamicEnergy

        print("-------------------- Estimation of ",self.name," --------------------")
        print("Tile readLatency: {:.2f}ns".format(self.Tile.readLatency/1e-9))
        print("Tile readDynamicEnergy: {:.2f}pJ".format(self.Tile.readDynamicEnergy * 1e12))
        print("readLatency: {:.2f}ns".format(self.readLatency/1e-9))
        print("readDynamicEnergy: {:.2f}pJ".format(self.readDynamicEnergy * 1e12))
        print("Buffer readLatency: {:.2f}ns".format(self.BufferreadLatency/1e-9))
        print("Buffer readDynamicEnergy: {:.2f}pJ".format(self.BufferreadDynamicEnergy * 1e12))
        print("IC readLatency: {:.2f}ns".format(self.ICreadLatency/1e-9))
        print("ICreadDynamicEnergy: {:.2f}pJ".format(self.ICreadDynamicEnergy * 1e12))
        print("Layer SubArray readLatency: {:.2f}ns".format(self.SubArrayreadLatency/1e-9))
        print("Layer SubArray readDynamicEnergy: {:.2f}pJ".format(self.SubArrayreadDynamicEnergy * 1e12))

