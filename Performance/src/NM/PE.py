import numpy as np
import neurosim
from Performance.src.Modules import calculate_col_resistance
from Performance.src.Configuration import configuration
from Performance.src.Modules import Wire



class PE():
    def __init__(self,input_param,tech,cell):
        self.conf = configuration()
        self.conf.settings_generation()
        # cong
        self.cell = cell
        self.input_param = input_param
        self.tech = tech
        self.wire = Wire(self.cell)
        
        #cut the weight matrix into different subarray
        if  self.conf.weightmapping == "Sign":
                if self.conf.signmapping == "NPsplit":
                    self.DigitPerWeight = int(np.ceil((self.conf.numBitWeight-1)/self.cell.cellBit))
                else:
                    self.DigitPerWeight = int(np.ceil((self.conf.numBitWeight-1)/self.cell.cellBit) + 1)
        else:
            self.DigitPerWeight = int(np.ceil(self.conf.numBitWeight / self.cell.cellBit))
        
        
        self.NumRows = self.conf.numRowNMPE
        self.NumCols = self.conf.numColNMPE
        self.Array= self.conf_neurosim_array(input_param,tech,cell,self.conf)
        if self.conf.weightmapping == "Sign" and self.conf.signmapping == "NPsplit":
            # if NPsplit is used for signed weight, two arrays are used to repesent the same value
            self.Array_n = self.conf_neurosim_array(input_param, tech, cell, self.conf)
            self.npAdder = neurosim.Adder(input_param, tech, cell)
        self.AdderTree = neurosim.AdderTree(input_param,tech,cell)
        
        if self.conf.inputshift!= 0:
            self.InshiftAdder = neurosim.Adder(input_param, tech, cell)
            self.Inshiftbuffer = neurosim.DFF(input_param, tech, cell)
        self.bufferInputCore =  neurosim.DFF(input_param,tech,cell)
        self.bufferOutputCore =  neurosim.DFF(input_param,tech,cell)
        self.bufferCoreRow = 32
        self.bufferCoreCol = 32
        self.busInput  = neurosim.Bus(input_param,tech,cell)
        self.busOutput = neurosim.Bus(input_param,tech,cell)

    def Configure(self):
        self.outputprecision = 0
        self.outputwidth = 0

        #=====================array setting======================
        self.Array.CalculateArea()
        self.outputprecision += self.Array.outputprecision
        self.outputwidth  += self.Array.outputwidth * self.conf.numRowNMPE * self.conf.numColNMPE
        self.outputwidth = int(self.outputwidth / self.conf.numRowNMPE)
        #extended array part
        #two array per weight matrix consider NP split case
        if self.conf.weightmapping == "Sign" and self.conf.signmapping == "NPsplit":
            self.Array_n.CalculateArea()
            self.outputprecision += 1
            self.npAdder.Configure( int(self.outputprecision), self.Array.outputwidth, self.conf.clkFreq )
        # if inputshit is not zero, the shiftinput will be applied to the weight.
        # as the shiftinput is the same across all input vectors, calculate once and saved
        # each columne output is different due to different weight
        if self.conf.inputshift!= 0:
            self.Inshiftbuffer.Configure(int(self.outputprecision)*self.conf.numColSubArray, self.conf.clkFreq)
            self.outputprecision += 1
            self.InshiftAdder.Configure( int(self.outputprecision), self.Array.outputwidth, self.conf.clkFreq )


        self.bufferInputCore.Configure(self.conf.numRowSubArray * self.conf.numBitInput , self.conf.clkFreq)
        self.bufferInputcoreNum = 1

        self.busInput.Configure(neurosim.BusMode.HORIZONTAL, self.NumRows, self.NumCols, 0, self.conf.numRowSubArray, self.Array.height, self.Array.width, self.conf.clkFreq)

        #addertree will add the rows together, reduce the outputwidth from circuit side.
        #due to duplication, different layer may require different outputwidth, which is not considered here
        
        self.AdderTree.Configure(self.conf.numRowNMPE, int(self.outputprecision), self.outputwidth, self.conf.clkFreq )
        self.outputprecision += np.log2(self.conf.numRowNMPE)

        self.busOutput.Configure(neurosim.BusMode.VERTICAL, self.NumRows,self.NumCols, 0, self.conf.numColSubArray, self.Array.height, self.Array.width, self.conf.clkFreq)

        self.bufferOutputCore.Configure(int(self.conf.numColSubArray / self.conf.numColMuxed * self.outputprecision), self.conf.clkFreq)
        self.bufferOutputcoreNum = 1

    def CalculateArea(self):
        singlearray_area = 0
        singlearray_area+=self.Array.usedArea
        Arraygroup_height=self.Array.height * self.NumRows
        Arraygroup_width = self.Array.width * self.NumCols
        if self.conf.weightmapping == "Sign" and self.conf.signmapping == "NPsplit":
            singlearray_area += self.Array_n.usedArea
            Arraygroup_width += self.Array_n.width * self.NumCols
            self.npAdder.CalculateArea(0, Arraygroup_width, neurosim.AreaModify.NONE)
            singlearray_area += self.npAdder.area
            Arraygroup_height += self.npAdder.height
        if self.conf.inputshift!= 0:
            self.InshiftAdder.CalculateArea(0, Arraygroup_width, neurosim.AreaModify.NONE)
            self.Inshiftbuffer.CalculateArea(0, Arraygroup_width, neurosim.AreaModify.NONE)
            Arraygroup_height += self.InshiftAdder.height
            Arraygroup_height += self.Inshiftbuffer.height
            singlearray_area +=  self.InshiftAdder.area
            singlearray_area +=  self.Inshiftbuffer.area

        self.bufferInputCore.CalculateArea(Arraygroup_height, 0, neurosim.AreaModify.NONE)
        self.bufferOutputCore.CalculateArea(0, Arraygroup_width, neurosim.AreaModify.NONE)
        
        self.AdderTree.CalculateArea(0, Arraygroup_width/self.NumCols, neurosim.AreaModify.NONE)

        self.busInput.CalculateArea(1, 1)
        self.busOutput.CalculateArea(1, 1)

        self.area = 0
        self.area += self.bufferInputCore.area * self.bufferInputcoreNum
        self.area += singlearray_area  * self.NumRows * self.NumCols
        self.area += self.bufferOutputCore.area * self.bufferOutputcoreNum
        self.area += self.AdderTree.area

        self.ArrayArea =  singlearray_area * self.NumRows * self.NumCols
        self.BufferArea = self.bufferInputCore.area * self.bufferInputcoreNum + self.bufferOutputCore.area * self.bufferOutputcoreNum
        self.DigitArea = self.AdderTree.area
        self.ICArea = 0
        if self.conf.inputshift!= 0:
            self.BufferArea += self.Inshiftbuffer.area * self.NumRows * self.NumCols
            self.DigitArea += self.InshiftAdder.area
        self.width = np.sqrt(self.area)
        self.height = self.area/self.width

        print("-------------------- Estimation of NM PE Area --------------------")
        print("PE area: ", self.area)
        print("Array area: ", self.ArrayArea," (", singlearray_area," x ", self.NumRows * self.NumCols,")")
        print("PE Buffer area: ", self.BufferArea)
        print("digit area: ", self.DigitArea)
        print("")

        if self.conf.printareaPE:
            print("================PE LEVEL AREA BREAKDOWN=================")
            print("PE area: ", self.area)
            print("Array area: ", self.ArrayArea," (", singlearray_area," x ", self.NumRows * self.NumCols,")")
            print("PE Buffer area: ", self.BufferArea)
            print("digit area: ", self.DigitArea)

    def CalculatePerformance(self,average_activityRowRead,input,inputshift,average_Condutance,weight,weight_n,average_dummy_conductance,speedup):
        #weight has the shape of [Fout,Fin], when mapped to the array, we put Fin for rows, thus need to be transposed
        self.bitop = 0
        self.numBitLoadin =  0
        self.numBitLoadout = 0
        self.OPnum = 0
        self.OPoutputwidth = 0
        self.OPoutputprecision = 0
        self.readLatency = 0
        self.readDynamicEnergy = 0
        self.BufferreadLatency = 0
        self.BufferreadDynamicEnergy = 0
        self.ICreadLatency = 0
        self.ICreadDynamicEnergy = 0
        self.DigitreadLatency = 0
        self.DigitreadDynamicEnergy = 0
        # add for subarray
        self.SubArrayreadLatency = 0
        self.SubArrayreadDynamicEnergy = 0
        # add for adc
        self.ADCreadLatency = 0
        self.ADCreadDynamicEnergy = 0
        
        
        weight_row = int(np.ceil(weight.shape[0]/self.conf.numRowSubArray))
        weight_col = int(np.ceil(weight.shape[1] * self.DigitPerWeight/self.conf.numColSubArray))
        trace_batch = input.shape[0]
        num_vector = input.shape[2]
        
        opoutputprecision = 0
        for batch_index in range(self.conf.batchsize):
            if self.conf.batchsize > trace_batch:
                raise ValueError("trace batchsize smaller than desired batchsize for test")

            for vector_index in range(1):
                input_vector = input[batch_index,:,vector_index]
                readDynamicEnergypervector, readLatencypervector, opoutputprecision,ADCreadLatencypervector,ADCreadDynamicEnergypervector = self.traced_Array_performance(average_activityRowRead,input_vector,average_Condutance,weight,
                                                                                                 weight_n,average_dummy_conductance,
                                                                                                 weight_col,weight_row)

                self.readLatency += num_vector*readLatencypervector
                self.readDynamicEnergy += num_vector*readDynamicEnergypervector
                self.ADCreadLatency += num_vector*ADCreadLatencypervector
                self.ADCreadDynamicEnergy +=num_vector*ADCreadDynamicEnergypervector

        self.readLatency = self.readLatency / (speedup[2] * speedup[3])
        self.ADCreadLatency = self.ADCreadLatency / (speedup[2] * speedup[3]) 
        
        self.SubArrayreadDynamicEnergy = self.readDynamicEnergy
        self.SubArrayreadLatency = self.readLatency
        
        
        self.OPoutputprecision = opoutputprecision
        self.OPoutputwidth = self.Array.outputwidth*weight_col*weight_row

        self.OPnum = self.conf.batchsize*num_vector*weight_col*self.conf.numColSubArray

        if weight_row > 1:
            self.AdderTree.CalculateLatency(self.conf.batchsize*num_vector*(weight_col*self.conf.numColSubArray/(self.Array.outputwidth*weight_col)), weight_row, 0)
            self.readLatency += self.AdderTree.readLatency
            self.AdderTree.CalculatePower(self.conf.batchsize*num_vector*(weight_col*self.conf.numColSubArray/(self.Array.outputwidth*weight_col)), weight_row)
            self.readDynamicEnergy += self.AdderTree.readDynamicEnergy
            self.OPoutputprecision = self.OPoutputprecision + np.log2(weight_row)
            self.OPoutputwidth = self.OPoutputwidth / weight_row

            self.DigitreadLatency += self.AdderTree.readLatency
            self.DigitreadDynamicEnergy = self.AdderTree.readDynamicEnergy

        self.numBitLoadin  = np.ceil(weight.shape[0]) * self.conf.numBitInput  * num_vector * self.conf.batchsize
        self.numBitLoadout = np.ceil(weight.shape[1]) * self.OPoutputprecision * num_vector * self.conf.batchsize

        #load from PE buffer to subarray buffer
        self.bufferInputCore.CalculateLatency(0, self.numBitLoadin/self.bufferInputCore.numDff)
        self.bufferInputCore.CalculatePower(self.numBitLoadin/self.bufferInputCore.numDff,self.bufferInputCore.numDff,0)
        self.bufferOutputCore.CalculateLatency(0, self.numBitLoadout/self.bufferOutputCore.numDff)
        self.bufferOutputCore.CalculatePower(self.numBitLoadout/self.bufferOutputCore.numDff,self.bufferOutputCore.numDff,0)
        self.readLatency += self.bufferInputCore.readLatency
        self.readLatency += self.bufferOutputCore.readLatency
        self.readDynamicEnergy += self.bufferInputCore.readDynamicEnergy
        self.readDynamicEnergy += self.bufferOutputCore.readDynamicEnergy

        self.BufferreadLatency  = self.bufferInputCore.readLatency +  self.bufferOutputCore.readLatency
        self.BufferreadDynamicEnergy  = self.bufferInputCore.readDynamicEnergy +  self.bufferOutputCore.readDynamicEnergy

        self.busInput.CalculateLatency(self.numBitLoadin / self.busInput.busWidth)
        self.busInput.CalculatePower(self.busInput.busWidth, self.numBitLoadin / self.busInput.busWidth)
        
        self.busOutput.CalculateLatency(self.numBitLoadout / (self.NumRows * self.busOutput.busWidth))
        self.busOutput.CalculatePower((self.NumRows * self.busOutput.busWidth), self.numBitLoadout / (self.NumRows * self.busOutput.busWidth))
        
        self.readLatency += self.busInput.readLatency
        self.readLatency += self.busOutput.readLatency
        self.readDynamicEnergy += self.busInput.readDynamicEnergy
        self.readDynamicEnergy += self.busOutput.readDynamicEnergy

        self.ICreadLatency  = self.busInput.readLatency +  self.busOutput.readLatency
        self.ICreadDynamicEnergy  = self.busInput.readDynamicEnergy +  self.busOutput.readDynamicEnergy


    def traced_Array_performance(self,average_activityRowRead,input_vector,average_Condutance,weight,weight_n,average_dummy_conductance,weight_col,weight_row):
        SubArrayreadDynamicEnergy = 0
        SubArrayreadLatency = 0
        opoutputprecision = 0

        ADCreadLatency = 0
        ADCreadDynamicEnergy = 0
        
        if self.conf.inputmapping == "Unsign":
            from Performance.src.integer2digit import dec2digit_unsign as i_dec2digit
        elif self.conf.inputmapping == "Sign":
            if self.conf.signmapping == "TwosComp":
                from Performance.src.integer2digit import dec2digit_sign_2s as i_dec2digit
            elif self.conf.signmapping == "NPsplit":
                from Performance.src.integer2digit import dec2digit_sign_np as i_dec2digit
            else:
                raise ValueError("Unknown signmapping")
        else:
            raise ValueError("Unknown weightmapping")

        input_bin_list_group, _ = i_dec2digit(input_vector,  self.conf.cycleBit, self.conf.numBitInput)

        if self.conf.inputmapping == "Sign" and self.conf.signmapping == "NPsplit":
            #if npsplit is used for input, fisrt pass positive bits and then negative
            input_bin_list = input_bin_list_group[0] + input_bin_list_group[1]
        else:
            input_bin_list = input_bin_list_group

        for array_col_index in range(weight_col):
            for array_row_index in range(weight_row):

                cinstart = array_row_index   * self.conf.numRowSubArray
                cinend = (array_row_index+1) * self.conf.numRowSubArray
                coutstart = array_col_index * int(self.conf.numColSubArray / self.DigitPerWeight)
                coutend =(array_col_index+1)* int(self.conf.numColSubArray / self.DigitPerWeight)

                input_vector = input_bin_list[0][cinstart:cinend]
                weight_subarray = weight[cinstart:cinend, coutstart:coutend]

                activated_row = average_activityRowRead * input_vector.shape[0]
                average_activityRowRead_Subarray = activated_row/self.conf.numRowSubArray

                self.Array.activityRowRead = average_activityRowRead_Subarray

                Res_Col, Dummy_Res_Col = calculate_col_resistance(average_activityRowRead_Subarray,average_Condutance, weight_subarray, average_dummy_conductance, self.conf, self.cell, self.wire,self.Array.resCellAccess, self.DigitPerWeight,self.conf.numColSubArray) 

                if self.Array.WeDummyCol:
                    Res_Col = Res_Col + Dummy_Res_Col
                             
                self.Array.activityRowWrite = 0.5
                self.Array.activityColWrite = 0.5
                self.Array.numWritePulse = self.Array.numRow * self.Array.numCol        
                self.Array.CalculateLatency(1e20, Res_Col, 0)
    
                #accumulate the subarray latency accross different input precision
                if SubArrayreadLatency < self.conf.numBitInput * self.Array.readLatency:
                    SubArrayreadLatency = self.conf.numBitInput * self.Array.readLatency
                    ADCreadLatency = self.conf.numBitInput * self.Array.readLatencyADC
        
                self.Array.CalculatePower(Res_Col)
        
                SubArrayreadDynamicEnergy += self.conf.numBitInput * self.Array.readDynamicEnergy
                ADCreadDynamicEnergy += self.conf.numBitInput * self.Array.readDynamicEnergyADC
        
                #count the bit op 1bit weight with 1bit input
                self.bitop += self.conf.numBitInput * 2 * self.conf.numRowSubArray * self.conf.numColSubArray 
                if weight_n == True:
                    SubArrayreadDynamicEnergy += self.conf.numBitInput * self.Array.readDynamicEnergy
                    ADCreadDynamicEnergy += self.conf.numBitInput * self.Array.readDynamicEnergyADC
                    
                    self.bitop += self.conf.numBitInput * 2 * self.conf.numRowSubArray * self.conf.numColSubArray 
                
        opoutputprecision = self.Array.outputprecision
        

        if self.conf.weightmapping == "Sign" and self.conf.signmapping == "NPsplit":
            self.npAdder.CalculateLatency(1e20,0,self.conf.numColMuxed)
            self.npAdder.CalculatePower(self.conf.numColMuxed,self.npAdder.numAdder)
            SubArrayreadLatency += self.npAdder.readLatency
            SubArrayreadDynamicEnergy += self.npAdder.readDynamicEnergy * weight_col * weight_row
            opoutputprecision += 1

        if self.conf.inputshift != 0:
            self.InshiftAdder.CalculateLatency(1e20,0,self.conf.numColMuxed)
            self.InshiftAdder.CalculatePower(self.conf.numColMuxed,self.InshiftAdder.numAdder)
            self.Inshiftbuffer.CalculateLatency(1e20,self.conf.numColMuxed)
            self.Inshiftbuffer.CalculatePower(self.conf.numColMuxed,self.Inshiftbuffer.numDff/self.conf.numColMuxed,0)
            SubArrayreadLatency += self.InshiftAdder.readLatency
            SubArrayreadDynamicEnergy += self.InshiftAdder.readDynamicEnergy * weight_col * weight_row
            SubArrayreadLatency += self.Inshiftbuffer.readLatency
            SubArrayreadDynamicEnergy += self.Inshiftbuffer.readDynamicEnergy * weight_col * weight_row
            opoutputprecision += 1

        return SubArrayreadDynamicEnergy,SubArrayreadLatency,opoutputprecision,ADCreadLatency,ADCreadDynamicEnergy

    def conf_neurosim_array(self,input_param,tech,cell,conf):
        Array = neurosim.SubArray(input_param, tech, cell)
        Array.conventionalSequential = 0
        Array.conventionalParallel = 1
        Array.BNNsequentialMode = 0
        Array.XNORsequentialMode = 0
        Array.numColMuxed = 0
        if conf.WeDummyCol == True:
            Array.WeDummyCol = 1
        else:
            Array.WeDummyCol = 0
            
                      
        Array.ADCmode = 0
        Array.numRow = conf.numRowSubArray
        Array.numCol = conf.numColSubArray
        Array.levelOutput = conf.levelOutput
        Array.numColMuxed = conf.numColMuxed  # // How many columns share 1 read circuit( for neuro mode with analog RRAM) or 1 S / A ( for memory mode or neuro mode with digital RRAM)
        Array.clkFreq = conf.clkFreq  # // Clock frequency
        Array.relaxArrayCellHeight = conf.relaxArrayCellHeight
        Array.relaxArrayCellWidth = conf.relaxArrayCellWidth
        Array.numReadPulse = conf.numBitInput
        Array.avgWeightBit = cell.cellBit
        Array.numCellPerSynapse = self.DigitPerWeight
        Array.SARADC = conf.SARADC
        Array.currentMode = conf.currentMode
        Array.validated = conf.validated
        Array.numReadCellPerOperationNeuro = conf.numColSubArray
        Array.numWriteCellPerOperationNeuro = conf.numColSubArray 
        Array.spikingMode = neurosim.SpikingMode.NONSPIKING
        Array.FPGA = 0
        Array.numReadCellPerOperationFPGA = 0
        Array.numWriteCellPerOperationFPGA = 0
        Array.numReadCellPerOperationMemory = 0
        Array.numWriteCellPerOperationMemory = 0
        Array.activityRowRead = 0
        Array.maxNumWritePulse = 0
    
    
        wireLengthRow = conf.wireWidth * 1e-9 * cell.heightInFeatureSize
        wireLengthCol = conf.wireWidth * 1e-9 * cell.widthInFeatureSize

        if conf.wireWidth == -1:
            unitLengthWireResistance = 1.0 # Use a small number to prevent numerical error for NeuroSim
            wireResistanceRow = 0
            wireResistanceCol = 0
        else :
            unitLengthWireResistance = conf.Rho / (conf.wireWidth * 1e-9 * conf.wireWidth * 1e-9 * conf.AR)
            wireResistanceRow = unitLengthWireResistance * wireLengthRow
            wireResistanceCol = unitLengthWireResistance * wireLengthCol

        Array.Configure(conf.numRowSubArray, conf.numColSubArray, unitLengthWireResistance)
        return Array



