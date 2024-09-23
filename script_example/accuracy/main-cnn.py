import os
import sys
import torch
import numpy as np
import random
import configparser
from Accuracy.src.Modules.CNN import datasets
from Accuracy.src.utils.seeds import set_seed
from datetime import datetime

set_seed(24)
# ====================init config path=========================
n = len(sys.argv)
if n == 2:
    os.environ['CONFIG'] = sys.argv[1]
else:
    os.environ['CONFIG'] = './Accuracy/config/vgg8/config_vgg_lsq_infer.ini'
# ====================init config path===========================

# ==================init the log================================
from Accuracy.src.utils import misc, make_path

logger = misc.logger.info
logdir = make_path.makepath_logdir()
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
misc.logger.init(logdir, 'train_log_' + current_time)
# ==================init the log================================

# ==================configuration================================
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
logger('load config from' + str(config.read(os.getenv('CONFIG'))))
logger('===============================configurations===============================')
for section in config.sections():
    logger('Section {:10s}'.format(section))
    for item in config[section]:
        logger('\t {:20s} : {}'.format(item, config[section][item]))
    # logger({section: dict(config[section])})
logger('===============================configurations===============================')
network = config['Network']['model']
dataset = config['Network']['dataset']
mode = config['Quantization']['mode']

batch_size = int(config['Training']['batch_size'])
gpuID = config['System']['gpu']
pretrained = config['Inference']['pretrained']
# ==================configuration================================


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuID


number_of_device = torch.cuda.device_count()
if number_of_device == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
from Accuracy.src.Modules.CNN import models
from Accuracy.src.Modules.CNN.train import Trainer
from Accuracy.src.Modules.CNN.Inference import Tester

# get the datasets used
train_loader, test_loader = datasets.load_datasets(dataset, batch_size)

# get the model
model = models.load_model(network, mode)

logger(model)
# get the trainer
train = Trainer(model.cuda(), train_loader, test_loader)
infer = Tester(model.cuda(), train_loader, test_loader)

# train the network
if pretrained != 'True':
    train.Training_with_Timer()

N = 1
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logger('start time: {}'.format(current_time))
for _ in range(N):
    infer._val()
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logger('finish time: {}'.format(current_time))
