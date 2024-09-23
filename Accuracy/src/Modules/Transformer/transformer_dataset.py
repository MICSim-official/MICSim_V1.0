import os
import configparser
import torch
from torch.utils.data.dataloader import Dataset, DataLoader
from transformers import BertTokenizer
from Accuracy.src.Modules.Transformer.glue_processor import *

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
model_path = config['Network']['model_path']
data_dir = config['Network']['data_path']
batch_size = int(config['Training']['batch_size'])
max_seq_length = int(config['Training']['max_seq_length'])


class GLUE_Dataset(Dataset):
    '''Load Data To Dataset'''
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

Task2Processor = {
    "CoLA": ColaProcessor,
    "SST-2": Sst2Processor,
    "MRPC": MrpcProcessor,
    "STS-B": StsbProcessor,
    "QQP": QqpProcessor,
    "MNLI": MnliProcessor,
    "QNLI": QnliProcessor,
    "RTE": QnliProcessor,
    "WNLI": QqpProcessor
}  

def collate_fn(data):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    input_ids, attention_mask, token_type_ids = [], [], []
    text_pair = None
    for x in data:
        if len(x) > 2:
            text_pair = x[1]            
        text = tokenizer(x[0], text_pair=text_pair, padding='max_length', truncation=True,\
             max_length=max_seq_length, return_tensors='pt')
        input_ids.append(text['input_ids'].squeeze().tolist())
        attention_mask.append(text['attention_mask'].squeeze().tolist())
        token_type_ids.append(text['token_type_ids'].squeeze().tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)

    label = torch.tensor([x[-1] for x in data])

    return input_ids, attention_mask, token_type_ids, label
    
def ToDataloader(task):
    '''Load data to train_dataloader, eval_dataloader and test_dataloader'''
    processor = Task2Processor[task]()
    data_path = os.path.join(data_dir, task)

    train_examples, eval_examples, test_examples = processor.get_train_examples(data_path),\
       processor.get_dev_examples(data_path), processor.get_test_examples(data_path)
    
    labels = processor.get_labels()
    train_data = GLUE_Dataset(train_examples)
    eval_data = GLUE_Dataset(eval_examples)
    test_data = GLUE_Dataset(test_examples)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    
    if task == "MNLI":
        eval_mis_examples, test_mis_examples = processor.get_dev_mismatched_examples(data_path), \
            processor.get_test_mismatched_examples(data_path)

        eval_mis = GLUE_Dataset(eval_mis_examples)
        test_mis = GLUE_Dataset(test_mis_examples)

        eval_mis_dataloader = DataLoader(eval_mis, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
        test_mis_dataloader = DataLoader(test_mis, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)

        return train_dataloader, eval_dataloader, eval_mis_dataloader, \
                     test_dataloader, test_mis_dataloader, labels

    return train_dataloader, eval_dataloader, test_dataloader, labels