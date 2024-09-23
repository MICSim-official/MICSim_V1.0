import numpy as np
import neurosim
from Performance.src.CM.PE import PE
from Performance.src.Configuration import configuration
from Performance.src.Modules import Wire


class Tile():
    def __init__(self,input_param,tech,cell):
        self.conf = configuration()
        self.conf.settings_generation()
        self.input_param = self.conf.input_param
        self.tech = tech
        self.cell = cell
        self.PE = PE(self.input_param,self.tech,self.cell)
        self.HTree =  neurosim.HTree(self.input_param,self.tech,self.cell)
        self.AdderTree =  neurosim.AdderTree(self.input_param,tech,self.cell)
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

    def Configure(self,TileNumRows,TileNumCols,PENumRows,PENumCols,SubarrayRows,SubarrayCols):
        self.NumRows = TileNumRows
        self.NumCols = TileNumCols
        self.PENumRows = PENumRows
        self.PENumCols = PENumCols
        self.SubarrayRows = SubarrayRows
        self.SubarrayCols = SubarrayCols
        
        self.PE.Configure(self.PENumRows, self.PENumCols,self.SubarrayRows,self.SubarrayCols)
        self.outputprecision = self.PE.outputprecision
        self.outputwidth = self.PE.outputwidth * self.NumCols * self.NumRows
        self.outputwidth = int(self.outputwidth/(self.NumRows))
        self.AdderTree.Configure(self.NumRows, int(self.outputprecision), self.outputwidth, self.conf.clkFreq )
        self.outputprecision += int(np.log2(self.NumRows))
        
        #on Tile input buffer
        self.bufferInputCore.Configure((self.bufferCoreRow*self.bufferCoreCol), self.bufferCoreRow,1, self.wire.unitLengthWireResistance, self.conf.clkFreq, False)

        self.bufferInputcoreNum = np.ceil(self.PENumRows  * self.SubarrayRows * self.NumRows * self.conf.numBitInput / (self.bufferCoreRow*self.bufferCoreCol))


        # on Tile output buffer
        self.bufferOutputCore.Configure((self.bufferCoreRow*self.bufferCoreCol), self.bufferCoreCol, 1, self.wire.unitLengthWireResistance, self.conf.clkFreq, False)
        self.bufferOutputcoreNum = np.ceil( self.outputwidth * self.outputprecision / (self.bufferCoreRow * self.bufferCoreCol))

        self.HTree.Configure(self.NumRows, self.NumCols, 0.1,  self.NumRows*self.SubarrayRows,self.conf.clkFreq)

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
        self.BufferArea += self.bufferOutputCore.area * self.bufferOutputcoreNum

        self.height = np.sqrt(self.area)
        self.width = self.area / self.height


        if self.conf.printareaTile:
            print("================PE LEVEL AREA BREAKDOWN=================")
            print("Tile area: ", self.area)
            print("Tile Buffer area: ", self.BufferArea,"( +",self.bufferOutputCore.area * self.bufferOutputcoreNum+self.bufferInputCore.area * self.bufferInputcoreNum,")")
            print("Tile IC area: ",     self.ICArea,"( +",self.HTree.area,")")
            print("Tile digit area: ",  self.DigitArea,"( +",self.AdderTree.area,")")


    def CalculatePerformance(self,average_activityRowRead,input,inputshift,average_Condutance, weight, weight_n, average_dummy_conductance,speedup):
        #input is the int input array, in the cm mode, it has the shape [B,Fin,H*W]
        #weight is the digit weight array, in the cm mode, it has the shape [Fout,Fin]
        #Fin/Fout means the Cin/Cout has been cut into tiles.
        #weight has the shape of [Fout,Fin], when mapped to the array, we put Fin for rows, thus need to be transposed
        
        # weight_n = true or none
        weight = weight.transpose(0,1)
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
        # add for write
        self.writeLatency = 0
        self.writeDynamicEnergy = 0
        self.SubArraywriteLatency = 0
        self.SubArraywriteDynamicEnergy = 0
        self.BufferwriteLatency = 0 
        self.BufferwriteDynamicEnergy = 0
        self.ICwriteLatency = 0
        self.ICwriteDynamicEnergy = 0

        self.readLatency = 0
        self.readDynamicEnergy = 0
        
        # add for adc
        self.ADCreadLatency = 0
        self.ADCreadDynamicEnergy = 0
            
        weight_row = int(np.ceil(weight.shape[0]/(self.PENumRows *self.SubarrayRows)))
        weight_col = int(np.ceil(weight.shape[1] * self.DigitPerWeight /(self.PENumCols*self.SubarrayCols)))

        NumPE_need = weight_row * weight_col

        
        Dup_col = 1 if int(self.NumCols/weight_col) == 0  else int(self.NumCols/weight_col)
        Dup_row = 1 if int(self.NumRows/weight_row) == 0  else int(self.NumRows/weight_row)
        
        DupPEnum = Dup_row * Dup_col
        
        for row_index in range(weight_row):
            for col_index in range(weight_col):

                cinstart = row_index   * (self.PENumRows*self.SubarrayRows)
                cinend = (row_index+1) * (self.PENumRows*self.SubarrayRows)
                coutstart = col_index * int((self.PENumCols*self.SubarrayCols)/self.DigitPerWeight)
                coutend =(col_index+1)* int((self.PENumCols*self.SubarrayCols)/self.DigitPerWeight)

                input_section = input[:,cinstart:cinend,:]
                weight_section = weight[cinstart:cinend, coutstart:coutend]


                if weight_n is not None:
                            weight_section_n = True 
                else:
                            weight_section_n = None

                self.PE.CalculatePerformance(average_activityRowRead,input_section, inputshift,average_Condutance, weight_section,weight_section_n,average_dummy_conductance,speedup)
                # 理解： PE/dup * numPEneed * dup
        
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

                # write
                if self.writeLatency < self.PE.writeLatency:
                    self.writeLatency = self.PE.writeLatency
                    self.SubArraywriteLatency  = self.PE.SubArraywriteLatency
                    self.BufferwriteLatency = self.PE.BufferwriteLatency
                    self.ICwriteLatency = self.PE.ICwriteLatency

                self.writeDynamicEnergy += self.PE.writeDynamicEnergy
                self.SubArraywriteDynamicEnergy += self.PE.SubArraywriteDynamicEnergy
                self.BufferwriteDynamicEnergy += self.PE.BufferwriteDynamicEnergy
                self.ICwriteDynamicEnergy += self.PE.ICwriteDynamicEnergy

        self.readLatency = self.readLatency / DupPEnum
        self.SubArrayreadLatency = self.SubArrayreadLatency / DupPEnum
        self.BufferreadLatency = self.BufferreadLatency /DupPEnum
        self.ICreadLatency = self.ICreadLatency / DupPEnum
        self.DigitreadLatency = self.DigitreadLatency/ DupPEnum
        # add for adc
        self.ADCreadLatency = self.ADCreadLatency/ DupPEnum
        
        
        self.OPoutputprecision = self.PE.OPoutputprecision
        
        
        if weight_row > 1:            
            self.AdderTree.CalculateLatency(self.conf.batchsize * num_vector * (self.conf.numColMuxed/self.DigitPerWeight)  ,weight_row,0)
            self.AdderTree.CalculatePower(self.conf.batchsize * num_vector * (self.conf.numColMuxed/self.DigitPerWeight),weight_row)
            self.readLatency += self.AdderTree.readLatency
            self.readDynamicEnergy += self.AdderTree.readDynamicEnergy
            self.DigitreadLatency += self.AdderTree.readLatency
            self.DigitreadDynamicEnergy += self.AdderTree.readDynamicEnergy

        self.OPoutputprecision += np.ceil(np.log2(weight_row))

        # write
        numBitLoadin_write = np.ceil(weight.shape[0] * weight.shape[1]) * self.conf.numBitWeight
        self.bufferInputCore.CalculateLatency(self.bufferInputCore.interface_width,
                                          np.ceil(numBitLoadin_write / (self.bufferInputCore.interface_width )),
                                          self.bufferInputCore.interface_width,
                                          np.ceil(numBitLoadin_write / (self.bufferInputCore.interface_width )))
        self.bufferInputCore.CalculatePower(self.bufferInputCore.interface_width,
                                          np.ceil(numBitLoadin_write / (self.bufferInputCore.interface_width)),
                                          self.bufferInputCore.interface_width,
                                          np.ceil(numBitLoadin_write / (self.bufferInputCore.interface_width)))
        
        self.writeLatency +=  (self.bufferInputCore.readLatency / np.minimum(self.bufferInputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width))
                            + self.bufferInputCore.writeLatency / np.minimum(self.bufferInputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width)))
        self.BufferwriteLatency +=  (self.bufferInputCore.readLatency / np.minimum(self.bufferInputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width))
                            + self.bufferInputCore.writeLatency / np.minimum(self.bufferInputcoreNum, np.floor(self.HTree.busWidth /self.bufferInputCore.interface_width)))
        self.writeDynamicEnergy +=  self.bufferInputCore.readDynamicEnergy + self.bufferInputCore.writeDynamicEnergy
        self.BufferwriteDynamicEnergy  +=  self.bufferInputCore.readDynamicEnergy + self.bufferInputCore.writeDynamicEnergy
        
        self.HTree.CalculateLatency(0, 0, 1, 1, self.PE.height, self.PE.width,(numBitLoadin_write)/self.HTree.busWidth )
        self.HTree.CalculatePower(0, 0, 1, 1, self.PE.height, self.PE.width,self.HTree.busWidth,(numBitLoadin_write)/self.HTree.busWidth )
        self.writeLatency +=  self.HTree.readLatency
        self.ICwriteLatency +=  self.HTree.readLatency
        self.writeDynamicEnergy +=  self.HTree.readDynamicEnergy
        self.ICwriteDynamicEnergy +=  self.HTree.readDynamicEnergy
        
        
        # read load input
        numBitToLoadIn = self.conf.numBitInput * input.shape[2] * weight.shape[0] * self.conf.batchsize 

        self.bufferInputCore.CalculateLatency(self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width )),
                                          self.bufferInputCore.interface_width,
                                          np.ceil(numBitToLoadIn / (self.bufferInputCore.interface_width )))
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

