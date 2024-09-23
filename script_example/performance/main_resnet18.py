from Performance.src.CNN.Network import Network
import time

# average_file_path = '/home/wangcong/projects/cimsimulator/average_files/ResNet18/LSQ4/CASE1/'
# average_file_path = '/home/wangcong/projects/cimsimulator/average_files/ResNet18/WAGE7/CASE1/'

average_file_path = '/home/wangcong/projects/cimsimulator/average_files/ResNet18/LSQ4/ADC/2bit/CASE2/'
layer_list = [
            [7,7,  3, 64,224,224, 2, 2, 0,average_file_path + 'layer1.csv', average_file_path + 'layer1weight_shift.csv', 'layer1', 'Conv'],
            
            [3,3, 64, 64, 56, 56, 1, 1, 0,average_file_path + 'group1_block1_conv1.csv', average_file_path +'group1_block1_conv1weight_shift.csv', 'group1_block1_conv1', 'Conv'],
            [3,3, 64, 64, 56, 56, 1, 1, 0,average_file_path + 'group1_block1_conv2.csv', average_file_path +'group1_block1_conv2weight_shift.csv', 'group1_block1_conv2', 'Conv'],
            [3,3, 64, 64, 56, 56, 1, 1, 0,average_file_path + 'group1_block2_conv1.csv', average_file_path +'group1_block2_conv1weight_shift.csv', 'group1_block2_conv1', 'Conv'],
            [3,3, 64, 64, 56, 56, 1, 1, 0,average_file_path + 'group1_block2_conv2.csv', average_file_path +'group1_block2_conv2weight_shift.csv', 'group1_block2_conv2', 'Conv'],
            
            [3,3, 64,128, 56, 56, 2, 2, 0,average_file_path + 'group2_block1_conv1.csv', average_file_path +'group2_block1_conv1weight_shift.csv', 'group2_block1_conv1', 'Conv'],
            [3,3,128,128, 28, 28, 1, 1, 0,average_file_path + 'group2_block1_conv2.csv', average_file_path +'group2_block1_conv2weight_shift.csv', 'group2_block1_conv2', 'Conv'],
            [1,1, 64,128, 56, 56, 2, 2, 0,average_file_path + 'group2_bypass.csv', average_file_path +'group2_bypassweight_shift.csv', 'group2_bypass', 'Conv'],
            [3,3,128,128, 28, 28, 1, 1, 0,average_file_path + 'group2_block2_conv1.csv', average_file_path +'group2_block2_conv1weight_shift.csv', 'group2_block2_conv1', 'Conv'],
            [3,3,128,128, 28, 28, 1, 1, 0,average_file_path + 'group2_block2_conv2.csv', average_file_path +'group2_block2_conv2weight_shift.csv', 'group2_block2_conv2', 'Conv'],
            
            [3,3,128,256, 28, 28, 2, 2, 0,average_file_path + 'group3_block1_conv1.csv', average_file_path +'group3_block1_conv1weight_shift.csv', 'group3_block1_conv1', 'Conv'],
            [3,3,256,256, 14, 14, 1, 1, 0,average_file_path + 'group3_block1_conv2.csv', average_file_path +'group3_block1_conv2weight_shift.csv', 'group3_block1_conv2', 'Conv'],
            [1,1,128,256, 28, 28, 2, 2, 0,average_file_path + 'group3_bypass.csv', average_file_path +'group3_bypassweight_shift.csv', 'group3_bypass', 'Conv'],
            [3,3,256,256, 14, 14, 1, 1, 0,average_file_path + 'group3_block2_conv1.csv', average_file_path +'group3_block2_conv1weight_shift.csv', 'group3_block2_conv1', 'Conv'],
            [3,3,256,256, 14, 14, 1, 1, 0,average_file_path + 'group3_block2_conv2.csv', average_file_path +'group3_block2_conv2weight_shift.csv', 'group3_block2_conv2', 'Conv'],
            
            [3,3,256,512, 14, 14, 2, 2, 0,average_file_path + 'group4_block1_conv1.csv', average_file_path +'group4_block1_conv1weight_shift.csv', 'group4_block1_conv1', 'Conv'],
            [3,3,512,512,  7,  7, 1, 1, 0,average_file_path + 'group4_block1_conv2.csv', average_file_path +'group4_block1_conv2weight_shift.csv', 'group4_block1_conv2', 'Conv'],
            [1,1,256,512, 14, 14, 2, 2, 0,average_file_path + 'group4_bypass.csv', average_file_path +'group4_bypassweight_shift.csv', 'group4_bypass', 'Conv'],
            [3,3,512,512,  7,  7, 1, 1, 0,average_file_path + 'group4_block2_conv1.csv', average_file_path +'group4_block2_conv1weight_shift.csv', 'group4_block2_conv1', 'Conv'],
            [3,3,512,512,  7,  7, 1, 1, 0,average_file_path + 'group4_block2_conv2.csv', average_file_path +'group4_block2_conv2weight_shift.csv', 'group4_block2_conv2', 'Conv'],
            
            [1,1,512,50, 1,  1, 1, 1, 0,average_file_path + 'last_layer.csv', average_file_path +'last_layerweight_shift.csv', 'last_layer', 'FC'],
]

def main_cnn():
    time_start = time.time()
    ResNet18 = Network(layer_list)
    ResNet18.Map()
    ResNet18.Configure()
    ResNet18.CalculateArea()
    ResNet18.CalculatePerformance()
    time_end = time.time()
    time_sum = time_end - time_start
    print("-------------------- Simulation Performance --------------------")
    print("Total Run-time of NeuroSim: {:.2f} seconds".format(time_sum))
    print("-------------------- Simulation Performance --------------------")

main_cnn()