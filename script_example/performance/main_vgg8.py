from Performance.src.CNN.Network import Network
import time

average_file_path = '/home/wangcong/projects/NeuroSim_Python_Version/average_file/vgg8/wage/average/'

layer_list = [
            [3, 3, 3,   128, 32, 32, 1, 1, 0,average_file_path + 'layer1.csv',None, 'layer1', 'Conv'],

            [3, 3, 128, 128, 32, 32, 1, 1, 0,average_file_path + 'layer2.csv',None, 'layer2', 'Conv'],

            [3, 3, 128, 256, 16, 16, 1, 1, 0,average_file_path + 'layer3.csv',None, 'layer3', 'Conv'],

            [3, 3, 256, 256, 16, 16, 1, 1, 0,average_file_path + 'layer4.csv',None, 'layer4', 'Conv'],

            [3, 3, 256, 512,  8,  8, 1, 1, 0,average_file_path + 'layer5.csv',None, 'layer5', 'Conv'],

            [3, 3, 512, 512,  8,  8, 1, 1, 0,average_file_path + 'layer6.csv',None, 'layer6', 'Conv'],

            [1, 1, 8192,1024, 1,  1, 1, 1, 0,average_file_path + 'layer7.csv',None, 'layer7', 'FC'],

            [1, 1, 1024,10,   1,  1, 1, 1, 0,average_file_path + 'layer8.csv',None, 'layer8', 'FC']]

def main_cnn():
    time_start = time.time()
    VGG8 = Network(layer_list)
    VGG8.Map()
    VGG8.Configure()
    VGG8.CalculateArea()
    
    VGG8.CalculatePerformance()
    time_end = time.time()
    time_sum = time_end - time_start
    print("-------------------- Simulation Performance --------------------")
    print("Total Run-time of CIMSim: {:.2f} seconds".format(time_sum))
    print("-------------------- Simulation Performance --------------------")

main_cnn()
