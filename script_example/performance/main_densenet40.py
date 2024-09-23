from Performance.src.CNN.Network  import Network
import time

average_file_path = '/home/wangcong/projects/NeuroSim_Python_Version/average_file/densenet40/wage/average/'

layer_list = [
            [3, 3, 3,   24, 32, 32, 1, 1, 0,average_file_path + 'layer1.csv', None, 'layer1', 'Conv'],
            [1, 1, 24,  48, 32, 32, 1, 1, 0,average_file_path + 'layer2.csv', None, 'layer2', 'Conv'],
            [3, 3, 48,  36, 32, 32, 1, 1, 0,average_file_path + 'layer3.csv', None, 'layer3', 'Conv'],
            [1, 1, 36,  48, 32, 32, 1, 1, 0,average_file_path + 'layer4.csv', None, 'layer4', 'Conv'],
            [3, 3, 48,  48, 32, 32, 1, 1, 0,average_file_path + 'layer5.csv', None, 'layer5', 'Conv'],
            [1, 1, 48,  48, 32, 32, 1, 1, 0,average_file_path + 'layer6.csv', None, 'layer6', 'Conv'],
            [3, 3, 48,  60, 32, 32, 1, 1, 0,average_file_path + 'layer7.csv', None, 'layer7', 'Conv'],
            [1, 1, 60,  48, 32, 32, 1, 1, 0,average_file_path + 'layer8.csv', None, 'layer8', 'Conv'],
            [3, 3, 48,  72, 32, 32, 1, 1, 0,average_file_path + 'layer9.csv', None, 'layer9', 'Conv'],
            [1, 1, 72,  48, 32, 32, 1, 1, 0,average_file_path + 'layer10.csv', None, 'layer10', 'Conv'],
            [3, 3, 48,  84, 32, 32, 1, 1, 0,average_file_path + 'layer11.csv', None, 'layer11', 'Conv'],
            [1, 1, 84,  48, 32, 32, 1, 1, 0,average_file_path + 'layer12.csv', None, 'layer12', 'Conv'],
            [3, 3, 48,  96, 32, 32, 1, 1, 0,average_file_path + 'layer13.csv', None, 'layer13', 'Conv'],
            [1, 1, 96,  48, 32, 32, 1, 1, 0,average_file_path + 'layer14.csv', None, 'layer14', 'Conv'],
            [1, 1, 48,  48, 16, 16, 1, 1, 0,average_file_path + 'layer15.csv', None, 'layer15', 'Conv'],
            [3, 3, 48,  60, 16, 16, 1, 1, 0,average_file_path + 'layer16.csv', None, 'layer16', 'Conv'],
            [1, 1, 60,  48, 16, 16, 1, 1, 0,average_file_path + 'layer17.csv', None, 'layer17', 'Conv'],
            [3, 3, 48,  72, 16, 16, 1, 1, 0,average_file_path + 'layer18.csv', None, 'layer18', 'Conv'],
            [1, 1, 72,  48, 16, 16, 1, 1, 0,average_file_path + 'layer19.csv', None, 'layer19', 'Conv'],
            [3, 3, 48,  84, 16, 16, 1, 1, 0,average_file_path + 'layer20.csv', None, 'layer20', 'Conv'],
            [1, 1, 84,  48, 16, 16, 1, 1, 0,average_file_path + 'layer21.csv', None, 'layer21', 'Conv'],
            [3, 3, 48,  96, 16, 16, 1, 1, 0,average_file_path + 'layer22.csv', None, 'layer22', 'Conv'],
            [1, 1, 96,  48, 16, 16, 1, 1, 0,average_file_path + 'layer23.csv', None, 'layer23', 'Conv'],
            [3, 3, 48,  108, 16, 16, 1, 1, 0,average_file_path + 'layer24.csv', None, 'layer24', 'Conv'],
            [1, 1, 108,  48, 16, 16, 1, 1, 0,average_file_path + 'layer25.csv', None, 'layer23', 'Conv'],
            [3, 3, 48,  120, 16, 16, 1, 1, 0,average_file_path + 'layer26.csv', None, 'layer26', 'Conv'],
            [1, 1, 120,  60, 16, 16, 1, 1, 0,average_file_path + 'layer27.csv', None, 'layer27', 'Conv'],
            [1, 1, 60,  48, 8, 8, 1, 1, 0,average_file_path + 'layer28.csv', None, 'layer28', 'Conv'],
            [3, 3, 48,  72, 8, 8, 1, 1, 0,average_file_path + 'layer29.csv', None, 'layer29', 'Conv'],
            [1, 1, 72,  48, 8, 8, 1, 1, 0,average_file_path + 'layer30.csv', None, 'layer30', 'Conv'],
            [3, 3, 48,  84, 8, 8, 1, 1, 0,average_file_path + 'layer31.csv', None, 'layer31', 'Conv'],
            [1, 1, 84,  48, 8, 8, 1, 1, 0,average_file_path + 'layer32.csv', None, 'layer32', 'Conv'],
            [3, 3, 48,  96, 8, 8, 1, 1, 0,average_file_path + 'layer33.csv', None, 'layer33', 'Conv'],
            [1, 1, 96,  48, 8, 8, 1, 1, 0,average_file_path + 'layer34.csv', None, 'layer34', 'Conv'],
            [3, 3, 48,  108, 8, 8, 1, 1, 0,average_file_path + 'layer35.csv', None, 'layer35', 'Conv'],
            [1, 1, 108,  48, 8, 8, 1, 1, 0,average_file_path + 'layer36.csv', None, 'layer36', 'Conv'],
            [3, 3, 48,  120, 8, 8, 1, 1, 0,average_file_path + 'layer37.csv', None, 'layer37', 'Conv'],
            [1, 1, 120,  48, 8, 8, 1, 1, 0,average_file_path + 'layer38.csv', None, 'layer38', 'Conv'],
            [3, 3, 48,  132, 8, 8, 1, 1, 0,average_file_path + 'layer39.csv', None, 'layer39', 'Conv'],
            [1, 1, 132,  10, 1, 1, 1, 1, 0,average_file_path + 'layer40.csv', None, 'layer40', 'FC'],

]

def main_cnn():
    time_start = time.time()
    DenseNet40 = Network(layer_list)
    DenseNet40.Map()
    DenseNet40.Configure()
    DenseNet40.CalculateArea()
    DenseNet40.CalculatePerformance()
    time_end = time.time()
    time_sum = time_end - time_start
    print("-------------------- Simulation Performance --------------------")
    print("Total Run-time of NeuroSim: {:.2f} seconds".format(time_sum))
    print("-------------------- Simulation Performance --------------------")

main_cnn()