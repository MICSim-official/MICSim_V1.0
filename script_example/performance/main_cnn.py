from Performance.src.CNN.Network import Network


average_file_path = '/home/wangcong/projects/MICSim_V1.0/average_files/WAGE3/Device/FeFET/'

layer_list = [
            [3, 3, 3,   128, 32, 32, 1, 1, 0,average_file_path + 'layer1.csv',average_file_path + 'layer1weight_shift.csv', 'layer1', 'Conv'],

            [3, 3, 128, 128, 32, 32, 1, 1, 0,average_file_path + 'layer2.csv', average_file_path + 'layer2weight_shift.csv','layer2', 'Conv'],

            [3, 3, 128, 256, 16, 16, 1, 1, 0,average_file_path + 'layer3.csv', average_file_path + 'layer3weight_shift.csv','layer3', 'Conv'],

            [3, 3, 256, 256, 16, 16, 1, 1, 0,average_file_path + 'layer4.csv', average_file_path + 'layer4weight_shift.csv','layer4', 'Conv'],

            [3, 3, 256, 512,  8,  8, 1, 1, 0,average_file_path + 'layer5.csv', average_file_path + 'layer5weight_shift.csv','layer5', 'Conv'],

            [3, 3, 512, 512,  8,  8, 1, 1, 0,average_file_path + 'layer6.csv', average_file_path + 'layer6weight_shift.csv','layer6', 'Conv'],

            [1, 1, 8192,1024, 1,  1, 1, 1, 0,average_file_path + 'layer7.csv', average_file_path + 'layer7weight_shift.csv','layer7', 'FC'],

            [1, 1, 1024,10,   1,  1, 1, 1, 0,average_file_path + 'layer8.csv', average_file_path + 'layer8weight_shift.csv','layer8', 'FC']]

VGG8 = Network(layer_list)
VGG8.Map()
VGG8.Configure()
VGG8.CalculateArea()
VGG8.CalculatePerformance()