import neurosim
import numpy as np
from Performance.src.CNN.Layer import Layer
from Performance.src.Configuration import configuration

DataPrecision = 8

class Network():
    def __init__(self,layer_list):
        self.conf = configuration()
        self.conf.settings_generation()
        self.input_param = self.conf.input_param
        self.tech = self.conf.tech
        self.StaticMVMCell = self.conf.StaticMVMCell
        self.DMVMKQCell = self.conf.DMVMKQCell
        self.DMVMPVCell = self.conf.DMVMPVCell
        
        self.globalBufferCore = neurosim.Buffer(self.input_param,self.tech,self.StaticMVMCell)
        self.GhTree =  neurosim.HTree(self.input_param,self.tech,self.StaticMVMCell) # global htree
        self.Layers = []
        self.layer_list = layer_list
        self.global_buffer_size = 0
        self.AdderArray =  neurosim.Adder(self.input_param,self.tech,self.StaticMVMCell)

        self.globalunitnum = 512

    def Map(self):
        #map the netwrok to the hardware chip. configure the floorplan
        print("User-defined Conventional Mapped Tile Storage Size: {}x{}".format(self.conf.numRowCMTile*self.conf.numRowCMPE*self.conf.numRowSubArray,self.conf.numColCMTile*self.conf.numColCMPE*self.conf.numColSubArray))
        print("User-defined Conventional PE Storage Size: {}x{}".format(self.conf.numRowCMPE*self.conf.numRowSubArray,self.conf.numColCMPE*self.conf.numColSubArray))
        print("User-defined Novel Mapped Tile Storage Size: {}x{}x{}".format(self.conf.numRowNMTile*self.conf.numColNMTile,self.conf.numRowNMPE*self.conf.numRowSubArray,self.conf.numColNMPE*self.conf.numColSubArray))
        print("User-defined SubArray Size: {}x{}".format(self.conf.numRowSubArray, self.conf.numColSubArray))

        for layer_i, layer_structure in enumerate(self.layer_list):
            if layer_structure[-1] == 'Conv' or layer_structure[-1] == 'FC':
                print(layer_structure)
                layer = Layer(self.input_param,self.tech,self.StaticMVMCell)
                layer.Map(layer_structure)
                self.Layers.append(layer)
                if self.global_buffer_size  < (layer_structure[3]*layer_structure[4]*layer_structure[5]*DataPrecision):
                    self.global_buffer_size = (layer_structure[3]*layer_structure[4]*layer_structure[5]*DataPrecision)
            elif layer_structure[-1] == 'MatmulKQ' or layer_structure[-1] == 'MatmulPV':
                if layer_structure[-1] == 'MatmulKQ':
                    layer = Layer(self.input_param,self.tech,self.DMVMKQCell)
                else:
                    layer = Layer(self.input_param,self.tech,self.DMVMPVCell)                       
                layer.Map(layer_structure)
                self.Layers.append(layer)
                if self.global_buffer_size  < (layer_structure[3]*layer_structure[4]*layer_structure[5]*DataPrecision):
                    self.global_buffer_size = (layer_structure[3]*layer_structure[4]*layer_structure[5]*DataPrecision)
                
        self.layer_type = []
        #collect types of layer used in the networks
        for l in self.layer_list:
            if  l[-1] not in self.layer_type:
                self.layer_type.append(l[-1])

    def Configure(self):
        self.TotalNMConvTiles = 0
        self.TotalCMConvTiles = 0
        self.TotalConvTiles = 0
        for layer in self.Layers:
            layer.Configure()
            if layer.NMmap:
                self.TotalNMConvTiles += layer.numTiles
            else:
                self.TotalCMConvTiles += layer.numTiles
        self.TotalConvTiles = self.TotalNMConvTiles  + self.TotalCMConvTiles

        self.NumRowOfTile = int(np.ceil(np.sqrt(self.TotalConvTiles)))
        self.NumColOfTile = int(np.ceil(self.TotalConvTiles / self.NumRowOfTile))
        
        self.GhTree.Configure(self.NumRowOfTile, self.NumColOfTile, 0.1, 4096, self.conf.clkFreq)

        self.globalBufferCore.Configure(128*128,128,1,1e7, self.conf.clkFreq, False)
        self.number_of_globalBufferCore = np.ceil(self.global_buffer_size/(128*128))

    def CalculateArea(self):
        self.area = 0
        self.bufferArea = 0
        self.HTreeArea = 0
        self.ArrayArea = 0
        self.layer_groups = []
        total_nm_height = 0
        total_cm_height = 0
        single_nm_width = 0
        single_cm_width = 0
        #print(self.layer_groups)
        for layer in self.Layers:
            layer.CalculateArea()
            self.area += layer.area
            self.bufferArea += layer.BufferArea
            self.HTreeArea += layer.ICArea
            self.ArrayArea += layer.ArrayArea
            if layer.NMmap:
                total_nm_height += layer.height
                single_nm_width = layer.width
            else:
                total_cm_height += layer.height
                single_cm_width = layer.width
            #print("outputprecision",layer.outputprecision*layer.OutputFeatureMapSize/( 8*layer.InputFeatureMapSize))


        Tile_H= np.maximum(total_nm_height / self.TotalNMConvTiles,total_cm_height / self.TotalCMConvTiles)
        Tile_W= np.maximum(single_nm_width, single_cm_width )

        self.TileArrayHeight = Tile_H * self.NumRowOfTile
        self.TileArrayWidth  = Tile_W * self.NumColOfTile



        # Top level global buffer
        self.globalBufferCore.CalculateArea(self.TileArrayHeight, -1, neurosim.AreaModify.NONE)
        self.area += self.globalBufferCore.area * self.number_of_globalBufferCore
        self.bufferArea += self.globalBufferCore.area * self.number_of_globalBufferCore

        self.GhTree.CalculateArea(Tile_H, Tile_W, 4)
        self.area += self.GhTree.area
        self.HTreeArea += self.GhTree.area


        print("-------------------- Estimation Chip Area --------------------")
        print("Chip area: ", self.area)
        print("Chip Array area: ", self.ArrayArea)
        print("Chip Buffer area: ", self.bufferArea)
        print("Chip IC area: ", self.HTreeArea)
        print("Chip Digit area: ", self.area-self.HTreeArea- self.bufferArea-self.ArrayArea)
        print("")
        print('Area total: ', self.area/1e-6, 'mm2')
        print('Area array: ', self.ArrayArea/1e-6, 'mm2')
        print('Area buffer: ', self.bufferArea/1e-6, 'mm2')
        print('Area ic: ', self.HTreeArea/1e-6, 'mm2')
        print('Area digital: ', (self.area-self.HTreeArea- self.bufferArea-self.ArrayArea)/1e-6, 'mm2')

    def CalculatePerformance(self):
        self.readLatency = 0
        self.readDynamicEnergy = 0
        self.SubArrayreadLatency = 0
        self.SubArrayreadDynamicEnergy = 0
        self.BufferreadLatency = 0
        self.BufferreadDynamicEnergy = 0
        self.ICreadLatency = 0
        self.ICreadDynamicEnergy = 0
        self.DigitreadLatency = 0
        self.DigitreadDynamicEnergy = 0
        
        # add for adc
        self.ADCreadLatency = 0
        self.ADCreadDynamicEnergy = 0

        end_tile_order = 0
        self.totalOP = 0

        for layer in self.Layers:
            layer_readLatency = 0
            layer_readDynamicEnergy = 0
            layer_BufferreadLatency = 0
            layer_BufferreadDynamicEnergy = 0
            layer_ICreadLatency = 0
            layer_ICreadDynamicEnergy = 0
            layer_SubArrayreadLatency = 0
            layer_SubArrayreadDynamicEnergy = 0

            buswidth_for_layer = np.floor(self.GhTree.busWidth / len(self.Layers))
            self.num_buswidth_parallel = np.ceil(buswidth_for_layer / self.globalBufferCore.interface_width)

            # load input feature map from global buffer

            numBitToLoadOut = layer.InputFeatureMapSize * layer.resend_rate * DataPrecision

            self.globalBufferCore.CalculateLatency(self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width)

            self.readLatency += self.globalBufferCore.readLatency / np.minimum(self.number_of_globalBufferCore,
                                                                                       self.num_buswidth_parallel)
            self.BufferreadLatency += self.globalBufferCore.readLatency / np.minimum(self.number_of_globalBufferCore,
                                                                                       self.num_buswidth_parallel)
            layer_readLatency = self.globalBufferCore.readLatency / np.minimum(self.number_of_globalBufferCore,
                                                                                       self.num_buswidth_parallel)
            layer_BufferreadLatency += self.globalBufferCore.readLatency / np.minimum(self.number_of_globalBufferCore,
                                                                                       self.num_buswidth_parallel)
            self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width)
            self.readDynamicEnergy    += self.globalBufferCore.readDynamicEnergy
            self.BufferreadDynamicEnergy  += self.globalBufferCore.readDynamicEnergy
            layer_readDynamicEnergy += self.globalBufferCore.readDynamicEnergy
            layer_BufferreadDynamicEnergy += self.globalBufferCore.readDynamicEnergy

            end_tile_order += layer.numTiles
            x_end = int(end_tile_order // self.NumRowOfTile)
            y_end = int(end_tile_order % self.NumRowOfTile)
            self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil( numBitToLoadOut / buswidth_for_layer))
            self.readLatency += self.GhTree.readLatency
            layer_readLatency += self.GhTree.readLatency
            self.ICreadLatency += self.GhTree.readLatency
            layer_ICreadLatency += self.GhTree.readLatency
            self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile,buswidth_for_layer, np.ceil( numBitToLoadOut/ buswidth_for_layer))
            self.readDynamicEnergy += self.GhTree.readDynamicEnergy
            layer_readDynamicEnergy += self.GhTree.readDynamicEnergy

            self.ICreadDynamicEnergy  += self.GhTree.readDynamicEnergy
            layer_ICreadDynamicEnergy += self.GhTree.readDynamicEnergy

            layer.CalculatePerformance()
            outputprecision = np.minimum(32,layer.OPoutputprecision)
            self.readLatency += layer.readLatency
            layer_readLatency +=  layer.readLatency
            self.BufferreadLatency += layer.BufferreadLatency
            layer_BufferreadLatency += layer.BufferreadLatency
            self.ICreadLatency += layer.ICreadLatency
            layer_ICreadLatency += layer.ICreadLatency
            self.SubArrayreadLatency += layer.SubArrayreadLatency
            layer_SubArrayreadLatency = layer.SubArrayreadLatency
            self.DigitreadLatency += layer.DigitreadLatency
            self.ADCreadLatency += layer.ADCreadLatency
            
            self.readDynamicEnergy += layer.readDynamicEnergy
            layer_readDynamicEnergy +=  layer.readDynamicEnergy
            self.SubArrayreadDynamicEnergy  += layer.SubArrayreadDynamicEnergy
            layer_SubArrayreadDynamicEnergy = layer.SubArrayreadDynamicEnergy
            self.BufferreadDynamicEnergy += layer.BufferreadDynamicEnergy
            layer_BufferreadDynamicEnergy += layer.BufferreadDynamicEnergy
            self.ICreadDynamicEnergy += layer.ICreadDynamicEnergy
            layer_ICreadDynamicEnergy += layer.ICreadDynamicEnergy
            self.DigitreadDynamicEnergy += layer.DigitreadDynamicEnergy
            self.ADCreadDynamicEnergy += layer.ADCreadDynamicEnergy

            numBitToLoadIn =  layer.OutputFeatureMapSize*outputprecision

            self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil(numBitToLoadIn/ buswidth_for_layer))
            self.readLatency += self.GhTree.readLatency
            layer_readLatency  += self.GhTree.readLatency
            self.ICreadLatency += self.GhTree.readLatency
            layer_ICreadLatency += self.GhTree.readLatency
            self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                       self.TileArrayWidth / self.NumColOfTile, buswidth_for_layer,
                                       np.ceil(numBitToLoadIn/ buswidth_for_layer))
            self.readDynamicEnergy  += self.GhTree.readDynamicEnergy
            layer_readDynamicEnergy += self.GhTree.readDynamicEnergy

            self.ICreadDynamicEnergy  += self.GhTree.readDynamicEnergy
            layer_ICreadDynamicEnergy += self.GhTree.readDynamicEnergy

            self.globalBufferCore.CalculateLatency(self.globalBufferCore.interface_width,
                                                   numBitToLoadIn / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width,
                                                   numBitToLoadIn / self.globalBufferCore.interface_width)
            self.readLatency += self.globalBufferCore.writeLatency / np.minimum(self.number_of_globalBufferCore,
                                                                             self.num_buswidth_parallel)
            layer_readLatency += self.globalBufferCore.writeLatency / np.minimum(self.number_of_globalBufferCore,
                                                                             self.num_buswidth_parallel)
            self.BufferreadLatency += self.globalBufferCore.writeLatency / np.minimum(self.number_of_globalBufferCore,
                                                                             self.num_buswidth_parallel)
            layer_BufferreadLatency += self.globalBufferCore.writeLatency / np.minimum(self.number_of_globalBufferCore,
                                                                             self.num_buswidth_parallel)
            self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width,
                                                 numBitToLoadIn / self.globalBufferCore.interface_width,
                                                 self.globalBufferCore.interface_width,
                                                 numBitToLoadIn / self.globalBufferCore.interface_width)
            self.readDynamicEnergy += self.globalBufferCore.writeDynamicEnergy
            layer_readDynamicEnergy += self.globalBufferCore.writeDynamicEnergy
            self.BufferreadDynamicEnergy += self.globalBufferCore.writeDynamicEnergy
            layer_BufferreadDynamicEnergy += self.globalBufferCore.writeDynamicEnergy

            self.totalOP += layer.OP


       
        print('-------------------- Summary --------------------')
        print('Total energy test:',(self.BufferreadDynamicEnergy+self.ICreadDynamicEnergy+self.SubArrayreadDynamicEnergy+self.DigitreadDynamicEnergy)*1e12,'pJ')
        print('self.BufferreadDynamicEnergy:',self.BufferreadDynamicEnergy*1e12,'pJ')
        print('self.ICreadDynamicEnergy:',self.ICreadDynamicEnergy*1e12,'pJ')
        print('self.SubArrayreadDynamicEnergy',self.SubArrayreadDynamicEnergy*1e12,'pJ')
        print('self.DigitreadDynamicEnergy:',self.DigitreadDynamicEnergy*1e12,'pJ')
        print('self.totalOP',self.totalOP)
        
        print('TOPS/W',float(self.totalOP)/1e12/self.readDynamicEnergy)
        print('TOPS/W/mm^2',float(self.totalOP)/1e12/self.readDynamicEnergy/(self.area*1e6))
        print('Throughput TOPS:',float(self.totalOP)/1e12/self.readLatency)
        print('Compute efficiency TOPS/mm^2:',float(self.totalOP)/(1e12*self.readLatency)/(self.area*1e6))
        
        print('Total Latency', self.readLatency/1e-9, 'ns')
        print('IC latency',(self.ICreadLatency)/1e-9,'ns')
        print('Buffer latency',(self.BufferreadLatency)/1e-9,'ns')
        
        print('Total Energy ', (self.readDynamicEnergy)/1e-12, 'pJ')
        
        print("Total Subarray read Latency: {:.2f}ns".format(self.SubArrayreadLatency/1e-9))
        print("Total Subarray Read Energy: {:.2f}pJ".format((self.SubArrayreadDynamicEnergy)/1e-12))
        
        print('ADC Latency',(self.ADCreadLatency)/1e-9, 'ns')
        print('ADC Energy', (self.ADCreadDynamicEnergy)/1e-12, 'pJ')


