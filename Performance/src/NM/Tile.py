import neurosim
import numpy as np
from Performance.src.NM.PE import PE
from Performance.src.Configuration import configuration
from Performance.src.Modules import Wire

class Tile():
    def __init__(self,input_param,tech,cell):
        self.conf = configuration()
        self.conf.settings_generation()
        self.input_param = self.conf.input_param
        self.tech = tech
        self.cell = cell
        self.NumRows = self.conf.numRowNMTile
        self.NumCols = self.conf.numColNMTile
        self.PE = PE(self.input_param,self.tech,self.cell)
        self.HTree =  neurosim.HTree(self.input_param,self.tech,self.cell)
        self.AdderTree =  neurosim.AdderTree(self.input_param,self.tech,self.cell)
        self.bufferInputCore =  neurosim.Buffer(self.input_param,self.tech,self.cell)
        self.bufferOutputCore = neurosim.Buffer(self.input_param,self.tech,self.cell)
        self.bufferCoreRow = 32
        self.bufferCoreCol = 32
        self.area = 0
        self.height = 0
        self.width = 0
        self.outputprecision = 0
        self.outputwidth = 0
        self.readLatency = 0
        self.readDynamicEnergy = 0
        
        self.wire = Wire(self.cell)
        
        if  self.conf.weightmapping == "Sign":
            if self.conf.signmapping == "NPsplit":
                self.DigitPerWeight = int(np.ceil((self.conf.numBitWeight-1)/self.cell.cellBit))
            else:
                self.DigitPerWeight = int(np.ceil((self.conf.numBitWeight-1)/self.cell.cellBit) + 1)
        else:
            self.DigitPerWeight = int(np.ceil(self.conf.numBitWeight / self.cell.cellBit))

    def Configure(self):
        self.PE.Configure()
        self.outputprecision = self.PE.outputprecision
        self.outputwidth = self.PE.outputwidth * self.NumCols * self.NumRows

        #in the NM mode, the tile level are added together, thus reduce the throughput
        # self.outputwidth = int(self.outputwidth/(self.NumCols * self.NumRows))
        
        self.AdderTree.Configure(int(self.NumCols * self.NumRows), int(self.outputprecision), self.outputwidth, self.conf.clkFreq )

        self.outputprecision +=  (np.ceil(np.log2(self.NumCols*self.NumRows)))
            
            
        #on Tile input buffer
        self.bufferInputCore.Configure((self.bufferCoreRow*self.bufferCoreCol), self.bufferCoreRow,1, self.wire.unitLengthWireResistance, self.conf.clkFreq, 0)
        # size is set to be able to hold the input for all PE(ARRAY) on Tile work at the same time (as different column will reuse the input)
        self.bufferInputcoreNum = np.ceil(self.conf.numRowSubArray * self.conf.numRowNMTile * self.conf.numBitInput / (self.bufferCoreRow*self.bufferCoreCol))

        # on Tile output buffer
        self.bufferOutputCore.Configure((self.bufferCoreRow*self.bufferCoreCol), self.bufferCoreCol, 1, self.wire.unitLengthWireResistance, self.conf.clkFreq, 0)
        self.bufferOutputcoreNum = np.ceil( self.outputwidth * self.outputprecision / (self.bufferCoreRow * self.bufferCoreCol))


        self.HTree.Configure(self.NumRows, self.NumCols, 0.1,  self.NumRows*self.conf.numRowSubArray,self.conf.clkFreq)

    def CalculateArea(self):
        self.PE.CalculateArea()

        self.bufferInputCore.CalculateArea(self.PE.height * self.NumRows, 0, neurosim.AreaModify.NONE)

        self.HTree.CalculateArea(self.PE.height, self.PE.width, 16)

        self.AdderTree.CalculateArea(0, self.PE.width * self.NumCols, neurosim.AreaModify.NONE)

        self.bufferOutputCore.CalculateArea(0, self.PE.width * self.NumCols, neurosim.AreaModify.NONE)

        self.area = 0
        self.area += self.PE.area * self.NumRows * self.NumCols

        self.ArrayArea  = self.PE.ArrayArea  * self.NumRows * self.NumCols
        self.BufferArea = self.PE.BufferArea * self.NumRows * self.NumCols
        self.ICArea     = self.PE.ICArea     * self.NumRows * self.NumCols
        self.DigitArea  = self.PE.DigitArea  * self.NumRows * self.NumCols

        self.area +=self.AdderTree.area
        self.DigitArea +=self.AdderTree.area

        self.area += self.HTree.area
        self.ICArea += self.HTree.area

        self.area += self.bufferInputCore.area * self.bufferInputcoreNum
        self.BufferArea += self.bufferInputCore.area * self.bufferInputcoreNum

        self.area += self.bufferOutputCore.area * self.bufferOutputcoreNum
        self.BufferArea  += self.bufferOutputCore.area * self.bufferOutputcoreNum

        self.height = np.sqrt(self.area)
        self.width = self.area / self.height

        print("-------------------- Estimation of NM Tile Area --------------------")
        print("Tile area: ", self.area)
        print("Tile Array area: ", self.ArrayArea," (", self.PE.ArrayArea," x ", self.NumRows * self.NumCols,")")
        print("Tile Buffer area: ", self.BufferArea)
        print("Tile IC area: ", self.ICArea)
        print("Tile Digit area: ", self.DigitArea)
        print("")

        if self.conf.printareaTile:
            print("================PE LEVEL AREA BREAKDOWN=================")
            print("Tile area: ", self.area)
            print("Tile Buffer area: ", self.BufferArea,"( +",self.bufferOutputCore.area * self.bufferOutputcoreNum+self.bufferInputCore.area * self.bufferInputcoreNum,")")
            print("Tile IC area: ",     self.ICArea,"( +",self.HTree.area,")")
            print("Tile digit area: ",  self.DigitArea,"( +",self.AdderTree.area,")")

    def CalculatePerformance(self,average_activityRowRead,input,inputshift,average_Condutance,weight,weight_n,average_dummy_conductance,speedup):

        weight = np.swapaxes(weight,0,1)

        input_fin = weight.shape[1]
        num_vector = input.shape[2]

        self.OPoutputprecision = 0
        self.SubArrayreadLatency = 0
        self.SubArrayreadDynamicEnergy = 0
        self.BufferreadLatency = 0
        self.BufferreadDynamicEnergy = 0
        self.ICreadLatency = 0
        self.ICreadDynamicEnergy = 0
        self.DigitreadLatency = 0
        self.DigitreadDynamicEnergy = 0

        self.readLatency = 0
        self.readDynamicEnergy = 0

        # add for adc
        self.ADCreadLatency = 0
        self.ADCreadDynamicEnergy = 0

        for k1_index in range(self.conf.numRowNMTile):
            for k2_index in range(self.conf.numColNMTile):
                start_point = k1_index*weight.shape[3]+k2_index
                input_section = input[:,start_point::weight.shape[2]*weight.shape[3],:]
                weight_section = weight[:,:,k1_index,k2_index]
                if weight_n is not None:
                    weight_section_n = True
                else:
                    weight_section_n = None
                self.PE.CalculatePerformance(average_activityRowRead,input_section, inputshift,average_Condutance, weight_section,weight_section_n,average_dummy_conductance,speedup)
                
                if self.readLatency < self.PE.readLatency:
                    self.readLatency = self.PE.readLatency
                    self.SubArrayreadLatency = self.PE.SubArrayreadLatency
                    self.BufferreadLatency = self.PE.BufferreadLatency
                    self.ICreadLatency = self.PE.ICreadLatency
                    self.DigitreadLatency = self.PE.DigitreadLatency
                    self.ADCreadLatency = self.PE.ADCreadLatency
                    
                self.readDynamicEnergy +=  self.PE.readDynamicEnergy
                self.SubArrayreadDynamicEnergy += self.PE.SubArrayreadDynamicEnergy
                self.BufferreadDynamicEnergy += self.PE.BufferreadDynamicEnergy
                self.ICreadDynamicEnergy += self.PE.ICreadDynamicEnergy
                self.DigitreadDynamicEnergy += self.PE.DigitreadDynamicEnergy
                self.ADCreadDynamicEnergy += self.PE.ADCreadDynamicEnergy

        self.OPoutputprecision = self.PE.OPoutputprecision

        self.AdderTree.CalculateLatency(self.conf.batchsize * num_vector * (self.conf.numColMuxed/self.DigitPerWeight),self.NumCols*self.NumRows,0)
        self.AdderTree.CalculatePower(self.conf.batchsize * num_vector * (self.conf.numColMuxed/self.DigitPerWeight),self.NumCols*self.NumRows)
        # self.AdderTree.CalculateLatency(self.conf.batchsize * num_vector * weight.shape[0] / self.AdderTree.numAdderTree,self.NumCols*self.NumRows,0)
        # self.AdderTree.CalculatePower(self.conf.batchsize * num_vector * weight.shape[0] / self.AdderTree.numAdderTree,self.NumCols*self.NumRows)

        self.readLatency += self.AdderTree.readLatency
        self.readDynamicEnergy += self.AdderTree.readDynamicEnergy
        self.DigitreadLatency += self.AdderTree.readLatency
        self.DigitreadDynamicEnergy += self.AdderTree.readDynamicEnergy

        self.OPoutputprecision += (np.ceil(np.log2(self.NumCols*self.NumRows)))

        # numBitToLoadIn = self.conf.numBitInput * input.shape[2] * input.shape[1] * self.conf.batchsize / self.NumRows
        numBitToLoadIn = self.conf.numBitInput * input.shape[2] * weight.shape[0] * self.conf.batchsize / self.NumRows

        self.bufferInputCore.CalculateLatency(self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width)),
                                          self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width)))
        self.bufferInputCore.CalculatePower(self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width)),
                                          self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width)))

        self.readLatency +=  (self.bufferInputCore.readLatency / np.minimum(self.bufferInputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width))
                            + self.bufferInputCore.writeLatency / np.minimum(self.bufferInputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width)))
        self.BufferreadLatency +=  (self.bufferInputCore.readLatency / np.minimum(self.bufferInputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width))
                            + self.bufferInputCore.writeLatency / np.minimum(self.bufferInputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width)))
        self.readDynamicEnergy +=  self.bufferInputCore.readDynamicEnergy + self.bufferInputCore.writeDynamicEnergy
        self.BufferreadDynamicEnergy  +=  self.bufferInputCore.readDynamicEnergy + self.bufferInputCore.writeDynamicEnergy

        # numBitToLoadOut = self.OPoutputprecision * input.shape[2] * weight.shape[0] * self.conf.batchsize
        numBitToLoadOut = self.OPoutputprecision * input.shape[2] * weight.shape[1] * self.conf.batchsize

        self.HTree.CalculateLatency(0, 0, 1, 1, self.PE.height, self.PE.width,(numBitToLoadIn+numBitToLoadOut)/self.HTree.busWidth )
        self.HTree.CalculatePower(0, 0, 1, 1, self.PE.height, self.PE.width,self.HTree.busWidth,(numBitToLoadIn+numBitToLoadOut)/self.HTree.busWidth )
        self.readLatency +=  self.HTree.readLatency
        self.ICreadLatency +=  self.HTree.readLatency
        self.readDynamicEnergy +=  self.HTree.readDynamicEnergy
        self.ICreadDynamicEnergy +=  self.HTree.readDynamicEnergy

        self.bufferOutputCore.CalculateLatency(self.bufferOutputCore.interface_width,
                                          np.ceil(numBitToLoadOut / (self.bufferOutputCore.interface_width)),
                                          self.bufferOutputCore.interface_width,
                                          np.ceil(numBitToLoadOut / (self.bufferOutputCore.interface_width)))
        self.bufferOutputCore.CalculatePower(self.bufferOutputCore.interface_width,
                                          np.ceil(numBitToLoadOut / (self.bufferOutputCore.interface_width)),
                                          self.bufferOutputCore.interface_width,
                                          np.ceil(numBitToLoadOut / (self.bufferOutputCore.interface_width)))
        self.readLatency +=  (self.bufferOutputCore.readLatency/np.minimum(self.bufferOutputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width)) \
                            + self.bufferOutputCore.writeLatency/np.minimum(self.bufferOutputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width)))
        self.BufferreadLatency +=  (self.bufferOutputCore.readLatency/np.minimum(self.bufferOutputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width)) \
                              + self.bufferOutputCore.writeLatency/np.minimum(self.bufferOutputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width)))
        self.readDynamicEnergy +=  self.bufferOutputCore.writeDynamicEnergy + self.bufferOutputCore.readDynamicEnergy
        self.BufferreadDynamicEnergy  +=  self.bufferOutputCore.writeDynamicEnergy+ self.bufferOutputCore.readDynamicEnergy


