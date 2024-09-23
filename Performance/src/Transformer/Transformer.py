import neurosim
import numpy as np
from Performance.src.Configuration import configuration
from Performance.src.Transformer.TransformerLayer import Layer

class Transformer():
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
        self.LinearWQs = []
        self.LinearWKs = []
        self.LinearWVs = []
        self.SequenceLayers = []
        self.ParallelLayer = []
        
        self.layer_list = layer_list
        self.global_buffer_size = 0
        self.AdderArray = neurosim.Adder(self.input_param,self.tech,self.StaticMVMCell)
        self.globalunitnum = 512

    def Map(self):
        #map the netwrok to the hardware chip. configure the floorplan
        for layer_i, layer_structure in enumerate(self.layer_list):
            if layer_structure[-1] == 'Conv' or layer_structure[-1] == 'FC':
                layer = Layer(self.input_param,self.tech,self.StaticMVMCell)
                layer.Map(layer_structure)
                self.Layers.append(layer)
                
                if layer_structure[-2] == 'linearWQ':
                    self.LinearWQs.append(layer)
                elif layer_structure[-2] == 'linearWK':
                    self.LinearWKs.append(layer)
                elif layer_structure[-2] == 'linearWV':
                    self.LinearWVs.append(layer) 
                else:
                    self.SequenceLayers.append(layer)

                if self.global_buffer_size  < (layer.InputFeatureMapSize*self.conf.numBitInput):
                    self.global_buffer_size = (layer.InputFeatureMapSize*self.conf.numBitInput)
            
            elif layer_structure[-1] == 'MatmulKQ' or layer_structure[-1] == 'MatmulPV':
                if layer_structure[-1] == 'MatmulKQ':
                    layer = Layer(self.input_param,self.tech,self.DMVMKQCell)
                else:
                    layer = Layer(self.input_param,self.tech,self.DMVMPVCell)                       
                layer.Map(layer_structure)
                self.Layers.append(layer)
                self.SequenceLayers.append(layer)
                if self.global_buffer_size  < (layer.InputFeatureMapSize*self.conf.numBitInput):
                    self.global_buffer_size = (layer.InputFeatureMapSize*self.conf.numBitInput)
                
        self.layer_type = []
        #collect types of layer used in the networks
        for l in self.layer_list:
            if  l[-1] not in self.layer_type:
                self.layer_type.append(l[-1])

    def Configure(self):
        
        self.TotalTiles = 0
        flagmatmulKQ = False
        flagmatmulPV = False
        for layer in self.Layers:
            layer.Configure()

            if layer.type == 'MatmulKQ' and flagmatmulKQ == True:
                continue
            
            if layer.type == 'MatmulPV' and flagmatmulPV == True:
                continue
            
            self.TotalTiles += layer.numTiles
            
            if layer.type == 'MatmulKQ':
                flagmatmulKQ = True

            if layer.type == 'MatmulPV':
                flagmatmulPV = True

        self.TotalConvTiles = self.TotalTiles

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
        self.dmm_area = 0
        self.layer_groups = []
        total_nm_height = 0
        total_cm_height = 0
        single_nm_width = 0
        single_cm_width = 0
        flagmatmulKQ = False
        flagmatmulPV = False
        
        for layer in self.Layers:        
            
            layer.CalculateArea()
            
            if layer.type == 'MatmulKQ' and flagmatmulKQ == True:
                continue
            
            if layer.type == 'MatmulPV' and flagmatmulPV == True:
                continue
            
            if layer.type == 'MatmulKQ' or layer.type == 'MatmulPV':
                self.dmm_area += layer.area
                
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
            
            if layer.type == 'MatmulKQ':
                flagmatmulKQ = True
                
            if layer.type == 'MatmulPV':
                flagmatmulPV = True


        Tile_H= (total_cm_height / self.TotalTiles)
        Tile_W= (single_cm_width )

        self.TileArrayHeight = Tile_H * self.NumRowOfTile
        self.TileArrayWidth  = Tile_W * self.NumColOfTile



        # Top level global buffer
        self.globalBufferCore.CalculateArea(self.TileArrayHeight, -1, neurosim.AreaModify.NONE)
        self.area += self.globalBufferCore.area * self.number_of_globalBufferCore
        self.bufferArea += self.globalBufferCore.area * self.number_of_globalBufferCore

        self.GhTree.CalculateArea(Tile_H, Tile_W, 4)
        self.area += self.GhTree.area
        self.HTreeArea += self.GhTree.area

        print('Area total', self.area/1e-6, 'mm2')
        print('Area for dynamic layer', self.dmm_area/1e-6, 'mm2', (self.dmm_area/self.area)*100,"%")
        print('Area array', self.ArrayArea/1e-6, 'mm2',(self.ArrayArea/self.area)*100,"%")
        print('Area buffer', self.bufferArea/1e-6, 'mm2',(self.bufferArea/self.area)*100,"%")
        print('Area htree', self.HTreeArea/1e-6, 'mm2',(self.HTreeArea/self.area)*100,"%")
        print('Area digital', (self.area-self.HTreeArea- self.bufferArea-self.ArrayArea)/1e-6, 'mm2')

    def CalculateOneLayerPerformance(self, layer, buswidth_for_layer, end_tile_order):
        
        LayerreadLatency = 0
        LayerreadDynamicEnergy = 0
        
        LayerBufferreadLatency = 0
        LayerBufferreadDynamicEnergy = 0
        
        LayerICreadLatency = 0
        LayerICreadDynamicEnergy = 0
        
        LayerSubArrayreadLatency = 0
        LayerSubArrayreadDynamicEnergy = 0
        
        LayerADCreadLatency = 0
        LayerADCreadDynamicEnergy = 0
        
        LayerDigitreadLatency = 0
        LayerDigitreadDynamicEnergy = 0
            
        # load input feature map from global buffer
        self.num_buswidth_parallel = np.floor(buswidth_for_layer / self.globalBufferCore.interface_width)
        if self.num_buswidth_parallel <= 1:
                self.num_buswidth_parallel = 1
                
        numBitToLoadOut = layer.InputFeatureMapSize * layer.resend_rate * self.conf.numBitInput
        
        # global buffer latency
        self.globalBufferCore.CalculateLatency(self.globalBufferCore.interface_width, numBitToLoadOut / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width, numBitToLoadOut / self.globalBufferCore.interface_width)
            
        LayerreadLatency += self.globalBufferCore.readLatency / np.minimum(self.number_of_globalBufferCore, self.num_buswidth_parallel)
        LayerBufferreadLatency += self.globalBufferCore.readLatency / np.minimum(self.number_of_globalBufferCore, self.num_buswidth_parallel)
        
        # global buffer energy
        self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width, numBitToLoadOut / self.globalBufferCore.interface_width,
                                                 self.globalBufferCore.interface_width, numBitToLoadOut / self.globalBufferCore.interface_width)
            
        LayerreadDynamicEnergy  += self.globalBufferCore.readDynamicEnergy
        LayerBufferreadDynamicEnergy  += self.globalBufferCore.readDynamicEnergy
        
        # htree    
        end_tile_order += layer.numTiles
        x_end = int(end_tile_order // self.NumRowOfTile)
        y_end = int(end_tile_order % self.NumRowOfTile)
        # htree latency 
        self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil( numBitToLoadOut / buswidth_for_layer))  
        LayerreadLatency += self.GhTree.readLatency
        LayerICreadLatency += self.GhTree.readLatency
        # htree energy    
        self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile,buswidth_for_layer, np.ceil( numBitToLoadOut/ buswidth_for_layer))
        LayerreadDynamicEnergy += self.GhTree.readDynamicEnergy
        LayerICreadDynamicEnergy  += self.GhTree.readDynamicEnergy
        
        # layer
        layer.CalculatePerformance()
        # layer latency
        LayerreadLatency += layer.readLatency
        LayerBufferreadLatency += layer.BufferreadLatency
        LayerICreadLatency += layer.ICreadLatency
        LayerSubArrayreadLatency += layer.SubArrayreadLatency
        LayerDigitreadLatency += layer.DigitreadLatency
        LayerADCreadLatency += layer.ADCreadLatency
        # layer energy
        LayerreadDynamicEnergy += layer.readDynamicEnergy
        LayerBufferreadDynamicEnergy += layer.BufferreadDynamicEnergy
        LayerICreadDynamicEnergy += layer.ICreadDynamicEnergy
        LayerSubArrayreadDynamicEnergy  += layer.SubArrayreadDynamicEnergy
        LayerDigitreadDynamicEnergy += layer.DigitreadDynamicEnergy
        LayerADCreadDynamicEnergy += layer.ADCreadDynamicEnergy
        
        # save output feature map from global buffer
        outputprecision = np.minimum(32,layer.OPoutputprecision)
        numBitToLoadIn =  layer.OutputFeatureMapSize*outputprecision
        
        # htree latency 
        self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil(numBitToLoadIn/ buswidth_for_layer))
        LayerreadLatency += self.GhTree.readLatency
        LayerICreadLatency += self.GhTree.readLatency
        # htree energy
        self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                       self.TileArrayWidth / self.NumColOfTile, buswidth_for_layer, np.ceil(numBitToLoadIn/ buswidth_for_layer))
        LayerreadDynamicEnergy  += self.GhTree.readDynamicEnergy
        LayerICreadDynamicEnergy += self.GhTree.readDynamicEnergy
        
        # buffer latency 
        self.globalBufferCore.CalculateLatency(self.globalBufferCore.interface_width, numBitToLoadIn / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width, numBitToLoadIn / self.globalBufferCore.interface_width)
        LayerreadLatency += self.globalBufferCore.writeLatency / np.minimum(self.number_of_globalBufferCore, self.num_buswidth_parallel)
        LayerBufferreadLatency += self.globalBufferCore.writeLatency / np.minimum(self.number_of_globalBufferCore, self.num_buswidth_parallel)
        # buffer energy
        self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width, numBitToLoadIn / self.globalBufferCore.interface_width,
                                                 self.globalBufferCore.interface_width, numBitToLoadIn / self.globalBufferCore.interface_width)
        LayerreadDynamicEnergy += self.globalBufferCore.writeDynamicEnergy
        LayerBufferreadDynamicEnergy += self.globalBufferCore.writeDynamicEnergy
        
        layerOP = layer.OP
        
        latency = [LayerreadLatency, LayerBufferreadLatency, LayerICreadLatency, LayerSubArrayreadLatency, LayerDigitreadLatency,LayerADCreadLatency]
        energy =  [LayerreadDynamicEnergy, LayerBufferreadDynamicEnergy, LayerICreadDynamicEnergy, LayerSubArrayreadDynamicEnergy,LayerDigitreadDynamicEnergy,LayerADCreadDynamicEnergy]
            
        return latency, energy, end_tile_order,layerOP

                    
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
        
        # add for write
        self.writeLatency = 0
        self.writeDynamicEnergy = 0
        # self.numBitLoadin_write = 0
        self.SubArraywriteLatency = 0
        self.SubArraywriteDynamicEnergy = 0
        self.BufferwriteLatency = 0 
        self.BufferwriteDynamicEnergy = 0
        self.ICwriteLatency = 0
        self.ICwriteDynamicEnergy = 0
        
        # add for adc
        self.ADCreadLatency = 0
        self.ADCreadDynamicEnergy = 0

        end_tile_order = 0
        self.totalOP = 0
        
        buswidth_for_layer = np.floor(self.GhTree.busWidth / len(self.Layers))
        self.num_buswidth_parallel = np.ceil(buswidth_for_layer / self.globalBufferCore.interface_width)
        
        for i, (LinearWQ, LinearWK, LinearWV) in enumerate(zip(self.LinearWQs, self.LinearWKs, self.LinearWVs)):
             
            WQreadLatency, WQreadDynamicEnergy, end_tile_order, WQOP = self.CalculateOneLayerPerformance(LinearWQ, buswidth_for_layer,end_tile_order)
            WKreadLatency, WKreadDynamicEnergy, end_tile_order, WKOP = self.CalculateOneLayerPerformance(LinearWK, buswidth_for_layer,end_tile_order)
            WVreadLatency, WVreadDynamicEnergy, end_tile_order, WVOP = self.CalculateOneLayerPerformance(LinearWV, buswidth_for_layer,end_tile_order)
            self.totalOP += WQOP + WKOP + WVOP
            
            # Linear WQ, WK, WV work in parallel
            parallel_readLatency = [max(values) for values in zip(WQreadLatency, WKreadLatency, WVreadLatency)]
            self.readLatency += parallel_readLatency[0]
            self.BufferreadLatency += parallel_readLatency[1]
            self.ICreadLatency += parallel_readLatency[2]
            self.SubArrayreadLatency += parallel_readLatency[3]
            self.DigitreadLatency += parallel_readLatency[4]
            self.ADCreadLatency += parallel_readLatency[5]
            
            parallel_readDynamicEnergy = [sum(values) for values in zip(WQreadDynamicEnergy, WKreadDynamicEnergy, WVreadDynamicEnergy)]
            self.readDynamicEnergy += parallel_readDynamicEnergy[0]
            self.BufferreadDynamicEnergy += parallel_readDynamicEnergy[1]
            self.ICreadDynamicEnergy += parallel_readDynamicEnergy[2]
            self.SubArrayreadDynamicEnergy += parallel_readDynamicEnergy[3]
            self.DigitreadDynamicEnergy += parallel_readDynamicEnergy[4]
            self.ADCreadDynamicEnergy += parallel_readDynamicEnergy[5]

        flag_tile_matmulKQ = False
        flag_tile_matmulPV = False

        for layer in self.SequenceLayers:
            
            layer_readLatency = 0
            layer_readDynamicEnergy = 0
            
            # load input feature map from global buffer
            numBitToLoadOut = layer.InputFeatureMapSize * layer.resend_rate * self.conf.numBitInput
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
            self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width,
                                                   numBitToLoadOut / self.globalBufferCore.interface_width)
            self.readDynamicEnergy    += self.globalBufferCore.readDynamicEnergy
            self.BufferreadDynamicEnergy  += self.globalBufferCore.readDynamicEnergy
            layer_readDynamicEnergy += self.globalBufferCore.readDynamicEnergy

            if layer.type == 'MatmulKQ' and flag_tile_matmulKQ == True:
                end_tile_order += 0
            elif layer.type == 'MatmulPV' and flag_tile_matmulPV == True:
                end_tile_order += 0
            else:
                end_tile_order += layer.numTiles
            
            if layer.type == 'MatmulKQ':
                flag_tile_matmulKQ = True
            
            if layer.type == 'MatmulPV':
                flag_tile_matmulPV == True
            
            x_end = int(end_tile_order // self.NumRowOfTile)
            y_end = int(end_tile_order % self.NumRowOfTile)
            
            self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil( numBitToLoadOut / buswidth_for_layer))
            self.readLatency += self.GhTree.readLatency
            layer_readLatency += self.GhTree.readLatency
            self.ICreadLatency += self.GhTree.readLatency
            self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile,buswidth_for_layer, np.ceil( numBitToLoadOut/ buswidth_for_layer))
            self.readDynamicEnergy += self.GhTree.readDynamicEnergy
            layer_readDynamicEnergy += self.GhTree.readDynamicEnergy
            self.ICreadDynamicEnergy  += self.GhTree.readDynamicEnergy
            
            layer.CalculatePerformance()
            
            outputprecision = np.minimum(32,layer.OPoutputprecision)
            self.readLatency += layer.readLatency
            layer_readLatency +=  layer.readLatency
            self.BufferreadLatency += layer.BufferreadLatency
            self.ICreadLatency += layer.ICreadLatency
            self.SubArrayreadLatency += layer.SubArrayreadLatency
            self.DigitreadLatency += layer.DigitreadLatency
            self.ADCreadLatency += layer.ADCreadLatency
            
            self.readDynamicEnergy += layer.readDynamicEnergy
            layer_readDynamicEnergy +=  layer.readDynamicEnergy
            self.SubArrayreadDynamicEnergy  += layer.SubArrayreadDynamicEnergy
            self.BufferreadDynamicEnergy += layer.BufferreadDynamicEnergy
            self.ICreadDynamicEnergy += layer.ICreadDynamicEnergy
            self.DigitreadDynamicEnergy += layer.DigitreadDynamicEnergy
            self.ADCreadDynamicEnergy += layer.ADCreadDynamicEnergy
            
            # asssign weight for write    
            if layer.type == 'MatmulKQ' or layer.type == 'MatmulPV':
                numBitToLoadOut_write = layer.WeightSize * self.conf.numBitWeight
                self.globalBufferCore.CalculateLatency(self.globalBufferCore.interface_width,numBitToLoadOut_write / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width,numBitToLoadOut_write / self.globalBufferCore.interface_width)
                self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width, numBitToLoadOut_write / self.globalBufferCore.interface_width,
                                                   self.globalBufferCore.interface_width, numBitToLoadOut_write / self.globalBufferCore.interface_width)

                self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil(numBitToLoadOut_write / buswidth_for_layer))
                self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile,buswidth_for_layer, np.ceil(numBitToLoadOut_write/ buswidth_for_layer))
                
            if layer.type == 'MatmulKQ':
                # the write latency of MatmulPV can be hidden
                self.writeLatency += layer.writeLatency
                self.SubArraywriteLatency += layer.SubArraywriteLatency
                self.BufferwriteLatency += layer.BufferwriteLatency
                self.ICwriteLatency += layer.ICwriteLatency

                self.writeLatency += self.globalBufferCore.readLatency / np.minimum(self.number_of_globalBufferCore, self.num_buswidth_parallel)
                self.BufferwriteLatency += self.globalBufferCore.readLatency / np.minimum(self.number_of_globalBufferCore, self.num_buswidth_parallel)
                self.writeLatency+= self.GhTree.readLatency
                self.ICwriteLatency += self.GhTree.readLatency

            if layer.type == 'MatmulKQ' or layer.type == 'MatmulPV':
                self.writeDynamicEnergy += layer.writeDynamicEnergy
                self.SubArraywriteDynamicEnergy += layer.SubArraywriteDynamicEnergy
                self.BufferwriteDynamicEnergy += layer.BufferwriteDynamicEnergy
                self.ICwriteDynamicEnergy  += layer.ICwriteDynamicEnergy 
                
                self.writeDynamicEnergy    += self.globalBufferCore.readDynamicEnergy
                self.BufferwriteDynamicEnergy  += self.globalBufferCore.readDynamicEnergy
                self.writeDynamicEnergy += self.GhTree.readDynamicEnergy
                self.ICwriteDynamicEnergy += self.GhTree.readDynamicEnergy    
                

            numBitToLoadIn =  layer.OutputFeatureMapSize*outputprecision

            self.GhTree.CalculateLatency(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                         self.TileArrayWidth / self.NumColOfTile, np.ceil(numBitToLoadIn/ buswidth_for_layer))
            self.readLatency += self.GhTree.readLatency
            layer_readLatency  += self.GhTree.readLatency
            self.ICreadLatency += self.GhTree.readLatency
            self.GhTree.CalculatePower(0, 0, x_end, y_end, self.TileArrayWidth / self.NumColOfTile,
                                       self.TileArrayWidth / self.NumColOfTile, buswidth_for_layer,
                                       np.ceil(numBitToLoadIn/ buswidth_for_layer))
            self.readDynamicEnergy  += self.GhTree.readDynamicEnergy
            layer_readDynamicEnergy += self.GhTree.readDynamicEnergy

            self.ICreadDynamicEnergy  += self.GhTree.readDynamicEnergy

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
            self.globalBufferCore.CalculatePower(self.globalBufferCore.interface_width,
                                                 numBitToLoadIn / self.globalBufferCore.interface_width,
                                                 self.globalBufferCore.interface_width,
                                                 numBitToLoadIn / self.globalBufferCore.interface_width)
            self.readDynamicEnergy += self.globalBufferCore.writeDynamicEnergy
            layer_readDynamicEnergy += self.globalBufferCore.writeDynamicEnergy
            self.BufferreadDynamicEnergy += self.globalBufferCore.writeDynamicEnergy

            self.totalOP += layer.OP
        

       
        print('============================== Summary ==============================')
        print('opnum:',(self.totalOP))
        print('Throughput TOPS:',float(self.totalOP)/1e12/ (self.readLatency + self.writeLatency))
        print('Compute efficiency TOPS/mm^2:',float(self.totalOP)/(1e12*(self.readLatency + self.writeLatency))/(self.area*1e6))
        print('TOPS/W', float(self.totalOP)/1e12/(self.readDynamicEnergy + self.writeDynamicEnergy))
        
        print('Total Latency', (self.readLatency + self.writeLatency)/1e-9, 'ns')
        print('Read Latency', self.readLatency/1e-9, 'ns')
        print('Write Latency', self.writeLatency/1e-9, 'ns')
        
        print('IC latency',(self.ICreadLatency+self.ICwriteDynamicEnergy)/1e-9,'ns')
        print('IC read latency',(self.ICreadLatency)/1e-9,'ns')
        print('IC write latency',(self.ICwriteLatency)/1e-9,'ns')
        print('Buffer latency',(self.BufferreadLatency+self.BufferwriteLatency)/1e-9,'ns')
        print('Buffer read latency',(self.BufferreadLatency)/1e-9,'ns')
        print('Buffer write latency',(self.BufferwriteLatency)/1e-9,'ns')
        
        print('Subarray read Latency', self.SubArrayreadLatency/1e-9, 'ns')
        print('Subarray write Latency', self.SubArraywriteLatency/1e-9, 'ns')
        print('ADC Latency',(self.ADCreadLatency)/1e-9, 'ns')
    
        
        print('Total Energy ', (self.readDynamicEnergy + self.writeDynamicEnergy)/1e-12, 'pJ')
        print('Read Energy', (self.readDynamicEnergy)/1e-12, 'pJ')
        print('Write Energy', (self.writeDynamicEnergy)/1e-12, 'pJ')
        
        print('Buffer DynamicEnergy',(self.BufferreadDynamicEnergy+self.BufferwriteDynamicEnergy)/1e-12,'pJ')
        print('Buffer Read DynamicEnergy', self.BufferreadDynamicEnergy/1e-12, 'pJ')
        print('Buffer Write DynamicEnergy', self.BufferwriteDynamicEnergy/1e-12, 'pJ')
        
        
        print('IC DynamicEnergy', (self.ICreadDynamicEnergy+self.ICwriteDynamicEnergy)/1e-12, 'pJ')
        print('IC Read DynamicEnergy', self.ICreadDynamicEnergy/1e-12, 'pJ')
        print('IC Write DynamicEnergy', self.ICwriteDynamicEnergy/1e-12, 'pJ')
        
        print('Subarray Energy', (self.SubArrayreadDynamicEnergy + self.SubArraywriteDynamicEnergy)/1e-12, 'pJ')
        print('Subarray Read Energy', (self.SubArrayreadDynamicEnergy)/1e-12, 'pJ')
        print('Subarray Write Energy', (self.SubArraywriteDynamicEnergy)/1e-12, 'pJ')
        print('ADC Energy', (self.ADCreadDynamicEnergy)/1e-12, 'pJ')
        
       