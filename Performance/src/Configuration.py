import neurosim

class configuration():
    def __init__(self):
        self.modeltype='cnn'
        self.numBitInput = 8
        self.numBitWeight = 8
        # "Unsign" "Sign"
        self.weightmapping = "Sign"
        self.inputmapping = "Sign"
        self.signmapping = "NPsplit"
        # TwosComp
        # NPsplit
        
        # self.Qmode = "WAGE"
        self.batchsize = 1
        # self.digitRef  = 1
        self.WeDummyCol = False
        # WeDummyCol = True, use dummy column
        
        # CASE 2 =0
        # this is decided by quantization algorithm
        self.inputshift = 0
        self.weightshift = 0

        self.accesstype = 1
        # 1: cell.accessType = CMOS_access
        # 2: cell.accessType = BJT_access
        # 3: cell.ahccessType = diode_accessself.clkFreq = 1e9
        # 4: cell.accessType = none_access (Crossbar Array)self.temp = 300

        self.transistortype = 1
        #1: inputParameter.transistorType = conventional

        self.deviceroadmap = 2
        # 1: inputParameter.deviceRoadmap = HPself.technode = 32
        # 2: inputParameter.deviceRoadmap = LSTPself.wireWidth = 56

        self.globalBufferType = False
        # false: register file
        # true: SRAM
        
        self.SARADC = 1
        self.currentMode = 1
        self.validated  = 1

        #manually defined FP size param
        # resnet18:64
        # self.numRowSubArray = 64
        # self.numColSubArray = 64
        self.numRowSubArray = 128
        self.numColSubArray = 128
        self.numRowNMPE = 4
        self.numColNMPE = 4
        # self.numRowCMPE = 4
        # self.numColCMPE = 4
        # densenet40
        self.numRowCMPE = 2
        self.numColCMPE = 2
        
        self.numRowNMTile = 3
        self.numColNMTile = 3
        self.numRowCMTile = 2
        self.numColCMTile = 2
        
        # FOR Transformer:
        self.numRowFCSubArray = 128
        self.numColFCSubArray = 128
        self.numRowFCPE = 3
        self.numColFCPE = 3
        self.numRowFCTile = 2
        # unsign npsplit
        self.numColFCTile = 4
        # 2compl
        # self.numColFCTile = 5
        
        self.numRowKQSubArray = 64
        self.numColKQSubArray = 64
        self.numRowKQPE = 1
        self.numColKQPE = 4
        self.numRowKQTile = 1
        self.numColKQTile = 4
        
        self.numRowPVSubArray = 128
        self.numColPVSubArray = 128
        self.numRowPVPE = 1
        self.numColPVPE = 4
        self.numRowPVTile = 1
        self.numColPVTile = 1
        
        

        #/*** option to relax subArray layout ***/
        self.relaxArrayCellHeight = 0           #// relax ArrayCellHeight or not
        self.relaxArrayCellWidth = 0            #// relax ArrayCellWidth or not



        self.numColMuxed = 8                    #// How many columns share 1 ADC (for eNVM and FeFET) or parallel SRAM
        self.quantizeADCtype = 0
        self.levelOutput = 256
        #// # of levels of the multilevelSenseAmp output, should be in 2^N forms; e.g. 32 levels --> 5-bit ADCself.heightInFeatureSizeSRAM = 10 #SRAM Cell height in feature size
        # TODO
        # SRAM = 1, ReRAM = 2
        # self.cellBit = 2                        #// precision of memory device 
        self.cycleBit = 1

        # *conventional hardware design options
        self.clkFreq = 1e9 # Clock frequency
        self.temp = 300 # Temperature(K)
        self.technode = 22 # Technology
        self.featuresize = 40e-9 # Wire width for subArray simulation
        self.wireWidth = 40 # wireWidth of the cell for Accuracy calculation

        
        #/*** parameters for analog synaptic devices ***/
        self.heightInFeatureSize1T1R = 4       #// 1T1R Cell height in feature size
        self.widthInFeatureSize1T1R = 12       #// 1T1R Cell width in feature size
        self.heightInFeatureSizeCrossbar = 2   #// Crossbar Cell height in feature size
        self.widthInFeatureSizeCrossbar = 2    #// Crossbar Cell width in feature size
        
        
        self.widthInFeatureSizeSRAM = 28 #SRAM Cell width in feature size
        self.heightInFeatureSizeSRAM = 10
        self.widthSRAMCellNMOS = 2 
        self.widthSRAMCellPMOS = 1 
        self.widthAccessCMOS = 1
        self.minSenseVoltage = 0.1
        self.readVoltage = 0.5          #   On-chip read voltage for memory cell
        self.readPulseWidth = 10e-9     #   read pulse width in sec
        self.accessVoltage = 1.1        #   Gate voltage for the transistor in 1T1R
        self.writeVoltage = 2           #   Enable level shifer if writeVoltage > 1.5V
        self.IR_DROP_TOLERANCE =0.25

        
        #################################################################
        self.StaticMVMmemcelltype = 3
        #1: cell.memCellType = Type::SRAM
        #2: cell.memCellType = Type::RRAM
        #3: cell.memCellType = Type::FeFET
        self.StaticMVMCellBit = 4 # SRAM=1, RRAM=1,2,4
        self.StaticMVMresistanceOn = 240*1e3                    #// Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
        self.StaticMVMresistanceOff = 240*1e5          #// Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
        self.StaticMVMresistanceAccess = self.StaticMVMresistanceOn*0.25
        
       ################################################################# 
        self.DMVMKQmemcelltype = 1  
        # memCellType for dynamic matrix multiplication is always SRAM in this version
        self.DMVMKQCellBit = 1 # SRAM‘s cell bit is always equal to 1 
        self.DMVMKQresistanceOn = 100e3                     #// Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
        self.DMVMKQresistanceOff = 100e3*100       #// Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
        self.DMVMKQresistanceAccess = self.DMVMKQresistanceOn*0.25
        
       ################################################################# 
        self.DMVMPVmemcelltype = 1
        # memCellType for dynamic matrix multiplication is always SRAM in this version
        self.DMVMPVCellBit = 1 # SRAM‘s cell bit is always equal to 1
        self.DMVMPVresistanceOn = 100e3                     #// Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
        self.DMVMPVresistanceOff = 100e3*100       #// Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity) 
        self.DMVMPVresistanceAccess = self.DMVMPVresistanceOn*0.25


        #Initialize interconnect wires
        wiretable = {
        175: [1.60, 2.20e-8], # for technode: 130
        110: [1.60, 2.52e-8], # for technode: 90
        105: [1.70, 2.68e-8], # for technode: 65
        80:  [1.70, 3.31e-8], # for technode: 45
        56:  [1.80, 3.70e-8], # for technode: 32
        40:  [1.90, 4.03e-8], # for technode: 22
        25:  [2.00, 5.08e-8], # for technode: 14
        18:  [2.00, 6.35e-8], # for technode: 7, 10
        }

        self.AR = wiretable[self.wireWidth][0]
        self.Rho = wiretable[self.wireWidth][1]

        
        ##################################################################
        #              TODO: these codes are in PE.py Linea368
        #              NOW: it is down in class wire
        # ##################################################################
        self.Rho *= (1 + 0.00451 * abs(self.temp - 300))

        self.printareaPE = False
        self.printareaTile = False
        self.printareaLayer = False

    def settings_generation(self):
        # set parameter
        input_param = neurosim.InputParameter()
        input_param.temperature =  self.temp
        input_param.transistorType = neurosim.TransistorType.conventional
        input_param.deviceRoadmap = neurosim.DeviceRoadmap.LSTP
        input_param.processNode = self.technode
        tech = neurosim.Technology()
        tech.Configure(self.technode, neurosim.DeviceRoadmap.LSTP, neurosim.TransistorType.conventional)
        
        
        # set static mvm subarray's cell parameter.
        StaticMVMCell = neurosim.MemCell()
        if self.StaticMVMmemcelltype == 1:
            StaticMVMCell.memCellType = neurosim.MemCellType.SRAM
            StaticMVMCell.widthInFeatureSize = self.widthInFeatureSizeSRAM
            StaticMVMCell.heightInFeatureSize = self.heightInFeatureSizeSRAM
            # new add
            StaticMVMCell.widthSRAMCellNMOS = self.widthSRAMCellNMOS 
            StaticMVMCell.widthSRAMCellPMOS = self.widthSRAMCellPMOS
            if self.accesstype == 1:
                # new add
                StaticMVMCell.widthAccessCMOS = self.widthAccessCMOS
            
        elif self.StaticMVMmemcelltype == 2 or self.StaticMVMmemcelltype == 3:
            if self.StaticMVMmemcelltype == 2:
                StaticMVMCell.memCellType = neurosim.MemCellType.RRAM   
            elif self.StaticMVMmemcelltype == 3:   
                StaticMVMCell.memCellType = neurosim.MemCellType.FeFET
                
            if self.accesstype == 1:
                
                StaticMVMCell.widthInFeatureSize = self.widthInFeatureSize1T1R
                StaticMVMCell.heightInFeatureSize = self.heightInFeatureSize1T1R
            else:
                StaticMVMCell.widthInFeatureSize = self.widthInFeatureSizeCrossbar
                StaticMVMCell.heightInFeatureSize = self.heightInFeatureSizeCrossbar    
                 
        else:  
            raise NotImplementedError("memCellType Error: Static MVM Cell only support SRAM, SRAM, FeFET now!")    
        # add new member cell bit
        StaticMVMCell.cellBit = self.StaticMVMCellBit
        StaticMVMCell.resistanceOn = self.StaticMVMresistanceOn
        StaticMVMCell.resistanceOff = self.StaticMVMresistanceOff
        StaticMVMCell.minSenseVoltage = self.minSenseVoltage
        StaticMVMCell.featureSize = self.featuresize
        StaticMVMCell.accessVoltage = self.accessVoltage
        StaticMVMCell.readPulseWidth = self.readPulseWidth
        StaticMVMCell.readVoltage = self.readVoltage
        StaticMVMCell.writeVoltage = self.writeVoltage
        StaticMVMCell.resistanceAccess = self.StaticMVMresistanceOn * self.IR_DROP_TOLERANCE 
        StaticMVMCell.resistanceAvg = (StaticMVMCell.resistanceOn + StaticMVMCell.resistanceOff) / 2
        
        # set dynamic mvm(query,key) subarray's cell parameter. 
        DMVMKQCell = neurosim.MemCell()
        if self.DMVMKQmemcelltype == 1:
            DMVMKQCell.memCellType = neurosim.MemCellType.SRAM
        else:
            raise NotImplementedError("memCellType Error: Dynamic MVM Cell only support SRAM now!")   
        
        DMVMKQCell.widthSRAMCellNMOS = self.widthSRAMCellNMOS 
        DMVMKQCell.widthSRAMCellPMOS = self.widthSRAMCellPMOS
        if self.accesstype == 1:
        # new add
            DMVMKQCell.widthAccessCMOS = self.widthAccessCMOS
        # add new member cell bit 
        DMVMKQCell.cellBit = self.DMVMKQCellBit 
        DMVMKQCell.widthInFeatureSize = self.widthInFeatureSizeSRAM
        DMVMKQCell.heightInFeatureSize = self.heightInFeatureSizeSRAM
        DMVMKQCell.resistanceOn = self.DMVMKQresistanceOn
        DMVMKQCell.resistanceOff = self.DMVMKQresistanceOff
        DMVMKQCell.minSenseVoltage = self.minSenseVoltage
        DMVMKQCell.featureSize = self.featuresize
        DMVMKQCell.accessVoltage = self.accessVoltage
        DMVMKQCell.readPulseWidth =self.readPulseWidth
        DMVMKQCell.readVoltage = self.readVoltage
        DMVMKQCell.writeVoltage  = self.writeVoltage
        DMVMKQCell.resistanceAccess = self.DMVMKQresistanceOn * self.IR_DROP_TOLERANCE
        DMVMKQCell.resistanceAvg = (DMVMKQCell.resistanceOn + DMVMKQCell.resistanceOff) / 2

        # set dynamic mvm(probs,value) subarray's cell parameter. 
        DMVMPVCell = neurosim.MemCell()
        if self.DMVMPVmemcelltype == 1:
            DMVMPVCell.memCellType = neurosim.MemCellType.SRAM
        else:
            raise NotImplementedError("memCellType Error: Dynamic MVM Cell only support SRAM now!")
        
        DMVMPVCell.widthSRAMCellNMOS = self.widthSRAMCellNMOS 
        DMVMPVCell.widthSRAMCellPMOS = self.widthSRAMCellPMOS
        if self.accesstype == 1:
        # new add
            DMVMPVCell.widthAccessCMOS = self.widthAccessCMOS 
        
        # add new member cell bit 
        DMVMPVCell.cellBit = self.DMVMPVCellBit  
        DMVMPVCell.widthInFeatureSize = self.widthInFeatureSizeSRAM
        DMVMPVCell.heightInFeatureSize = self.heightInFeatureSizeSRAM
        DMVMPVCell.resistanceOn = self.DMVMPVresistanceOn
        DMVMPVCell.resistanceOff = self.DMVMPVresistanceOff
        DMVMPVCell.minSenseVoltage = self.minSenseVoltage
        DMVMPVCell.featureSize = self.featuresize
        DMVMPVCell.accessVoltage = self.accessVoltage 
        DMVMPVCell.readPulseWidth = self.readPulseWidth
        DMVMPVCell.readVoltage = self.readVoltage
        DMVMPVCell.writeVoltage = self.writeVoltage
        DMVMPVCell.resistanceAccess = self.DMVMPVresistanceOn * self.IR_DROP_TOLERANCE
        DMVMPVCell.resistanceAvg = (DMVMPVCell.resistanceOn + DMVMPVCell.resistanceOff) / 2

        # pass member to class
        self.input_param = input_param
        self.tech = tech
        self.StaticMVMCell = StaticMVMCell
        self.DMVMKQCell = DMVMKQCell
        self.DMVMPVCell = DMVMPVCell