from Performance.src.Transformer.Transformer import Transformer
import time

batch_size =1
hidden_size=768
attention_head_size=64
sequence_length= 512
num_attention_heads= 12

average_file_path='./average_files/q8bert/benchmark/FeFET/'

Bert = []
for i in range(12):
    BertLayer = [
        [batch_size, hidden_size, attention_head_size, sequence_length, num_attention_heads, average_file_path + f'BertLayer_{i}_selfattention_weight_query_layer.csv',average_file_path + f'BertLayer_{i}_selfattention_weight_query_layerweight_shift.csv', 'linearWQ', 'FC'],
        [batch_size, hidden_size, attention_head_size, sequence_length, num_attention_heads, average_file_path + f'BertLayer_{i}_selfattention_weight_key_layer.csv',average_file_path + f'BertLayer_{i}_selfattention_weight_key_layerweight_shift.csv', 'linearWK','FC'],
        [batch_size, hidden_size, attention_head_size, sequence_length, num_attention_heads, average_file_path + f'BertLayer_{i}_selfattention_weight_value_layer.csv', average_file_path + f'BertLayer_{i}_selfattention_weight_value_layerweight_shift.csv','linearWV', 'FC'],
        [batch_size, hidden_size, attention_head_size, sequence_length, num_attention_heads, average_file_path + f'BertLayer_{i}_selfattention_MatmulKQ.csv', None,'MatmulKQ', 'MatmulKQ'],
        [batch_size, hidden_size, attention_head_size, sequence_length, num_attention_heads, average_file_path + f'BertLayer_{i}_selfattention_MatmulPV.csv', None, 'MatmulPV', 'MatmulPV'],
        [batch_size, hidden_size, attention_head_size, sequence_length, num_attention_heads, average_file_path + f'BertLayer_{i}_selfoutput_layer.csv',average_file_path + f'BertLayer_{i}_selfoutput_layerweight_shift.csv', 'selfoutput', 'FC']
    ]
    Bert.append(BertLayer)

Bert = [item for sublist in Bert for item in sublist]

time_start = time.time()
TEST = Transformer(Bert)
TEST.Map()
TEST.Configure()
TEST.CalculateArea()
TEST.CalculatePerformance()
time_end = time.time()
time_sum = time_end - time_start
print("-------------------- Simulation Performance --------------------")
print("Total Run-time of CIMSim: {:.2f} seconds".format(time_sum))
print("-------------------- Simulation Performance --------------------")
