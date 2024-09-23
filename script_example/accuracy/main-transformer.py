import os
import sys
import torch
import configparser
from datetime import datetime
from transformers import BertConfig, AdamW
from Accuracy.src.utils.seeds import set_seed

set_seed(24)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================init config path=========================
n = len(sys.argv)
if n == 2:
    os.environ['CONFIG'] = sys.argv[1]
else:
    os.environ['CONFIG'] = './Accuracy/config/transformer/config_stt2_bert_q8bert_infer.ini'
# ====================init config path===========================

# ==================init the log================================
from Accuracy.src.utils import misc, make_path

Logger = misc.logger.info
logdir = make_path.makepath_logdir()
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
misc.logger.init(logdir, 'train_log_' + current_time)
# ==================init the log================================

# ==================configuration================================
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
Logger('load config from' + str(config.read(os.getenv('CONFIG'))))
Logger('===============================configurations===============================')
for section in config.sections():
    Logger('Section {:10s}'.format(section))
    for item in config[section]:
        Logger('\t {:20s} : {}'.format(item, config[section][item]))

Logger('===============================configurations===============================')

network = config['Network']['model']
dataset = config['Network']['dataset']
task = config['Network']['task']
model_path = config['Network']['model_path']
finetuned = config['Inference']['finetuned']
finetuned_model = config['Inference']['finetuned_model'] 
mode = config['Quantization']['mode'] 

from Accuracy.src.Modules.Transformer.transformer_dataset import ToDataloader

if task == "MNLI":
    train_dataloader, eval_dataloader, eval_mis_dataloader, \
                     test_dataloader, test_mis_dataloader, labels = ToDataloader(task)
else:
    train_dataloader, eval_dataloader, test_dataloader, labels = ToDataloader(task)

if network == 'BERT':
    if mode == 'FloatingPoint':
        from Accuracy.src.Network.BERT.FloatingPoint.modeling_bert import BertForSequenceClassification 
    elif mode == 'IBERT':
        from Accuracy.src.Network.BERT.IBERT.modeling_bert import BertForSequenceClassification
    elif mode == 'Q8BERT':
        from Accuracy.src.Network.BERT.Q8BERT.modeling_bert import BertForSequenceClassification 

    
BERTconfig = BertConfig.from_pretrained(model_path)
BERTconfig.num_labels = len(labels) if type(labels) != float else 1
if finetuned == 'True':
    model = BertForSequenceClassification.from_pretrained(finetuned_model) 
else:
    model = BertForSequenceClassification.from_pretrained(model_path, config=BERTconfig)
model.to(device)

Logger(model)
lr = float(config['Training']['learning_rate'])
numepoch = int(config['Training']['numepoch'])
optimizer = AdamW(model.parameters(), lr=lr)

from Accuracy.src.Modules.Transformer.trainer import Trainer
from Accuracy.src.Modules.Transformer.tester import Tester


if task == "MNLI":
    trainer = Trainer(model,train_dataloader,eval_dataloader,
                  optimizer, eval_mis_dataloader)
else:
    trainer = Trainer(model,train_dataloader,eval_dataloader,
                  optimizer)

tester = Tester(model, eval_dataloader)
if finetuned != 'True':
    trainer.Training()


N = 1
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
Logger('start time: {}'.format(current_time))
for _ in range(N):
    tester._val()
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
Logger('finish time: {}'.format(current_time))

del model
torch.cuda.empty_cache()