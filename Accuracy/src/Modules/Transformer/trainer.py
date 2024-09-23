import os
import configparser
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from Accuracy.src.utils import  misc,make_path

logger = misc.logger.info
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
train_log_interval = int(config['Training']['train_log_interval'])
val_log_interval   = int(config['Training']['val_log_interval'])
numepoch = int(config['Training']['numepoch'])
lr = float(config['Training']['learning_rate'])
model_path = config['Network']['model_path']
task = config['Network']['task']
decreasing_lr = config['Training']['decreasing_lr']
mode = config['Quantization']['mode'] 

saved_model_path = model_path +'/' + task + '/' + mode +'/' +'4bit/'
if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
            
'''
 task_types = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification"
    }
    
'''            
            
class Trainer:
    def __init__(self,model,train_dataloader,eval_dataloader,optimizer,eval_mis_dataloader=None):
        self.model = model
        self.decreasing_lr = list(map(int, decreasing_lr.split(',')))
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_mis_dataloader = eval_mis_dataloader
        self.optimizer = optimizer
        self.task = task
        self.saved_model_path = saved_model_path
        self.best_metric = 0
        
    def Training(self):
        for epoch in range(numepoch):
            # train(self.task, self.train_dataloader, self.model, self.optimizer)
            self._update_one_epoch(epoch)
            metric = self._val(epoch, self.eval_dataloader)
            # metric = eval(self.task, self.eval_dataloader, self.model) 
            if self.task == "MNLI":
                dis_metric = self._val(epoch, self.eval_mis_dataloader)
                # dis_metric = eval(self.task, self.eval_mis_dataloader, self.model)
            if metric > self.best_metric:
                self.best_metric = metric
                logger("save best performance model to ")
                logger(self.saved_model_path)
                self.model.save_pretrained(self.saved_model_path)
                
            if epoch in self.decreasing_lr:
                self.optimizer.param_groups[0]['lr'] *= 0.5
    
    def _update_one_epoch(self,epoch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model.zero_grad()
        self.model.train() 
        y_l, y_p = [], []
        for batch_idx, batch in enumerate(self.train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            ## outputs ==> loss, logits, hidden_states, attentions
            outputs = self.model(**inputs)
            loss = outputs[0]
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
        
            if batch_idx % train_log_interval == 0 and batch_idx >= 0:
                acc, loss_num, train_len = 0, 0, 0
                logits = outputs[1]
                if self.task != "STS-B":
                    y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
                else:
                    y_pred = logits.detach().cpu().numpy()
            
                labels = inputs["labels"]
                labels = labels.detach().cpu().numpy()
            
                y_l += list(labels)
                y_p += list(y_pred)

                loss_num += loss.item()
                train_len += len(labels)

                if self.task == "STS-B":
                    y_p = [float(i) for i in y_p]
                    pearson_corr = pearsonr(y_p, y_l)[0] 
                    spearman_corr = spearmanr(y_p, y_l)[0]

                    logger('Train Epoch: {} [{}/{}] Loss: {:.4f} Pearson: {:.4f} Spearman: {:.4f}'
                        .format(epoch, epoch, numepoch-1, (loss_num / train_len), pearson_corr, spearman_corr))
                else:
                    acc = sum(np.array(y_l) == np.array(y_p)) / len(y_l)
                    logger('Train Epoch: {} [{}/{}] Loss: {:.4f} Acc: {:.4f} lr:{:8f}'
                           .format(epoch, epoch, numepoch-1,(loss_num / train_len), acc, self.optimizer.param_groups[0]['lr'] ))
                    
                    
    def _val(self, epoch, dataloader):
        logger("===================== testing phase =====================")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        y_l, y_p = [], []
        with torch.no_grad():
            # tq = tqdm(self.eval_dataloader)
            for step, batch in enumerate(dataloader):
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }

                ## outputs ==> loss, logits, hidden_states, attentions
                outputs = self.model(**inputs)
                logits = outputs[1]
                if self.task != "STS-B":
                    y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
                else:
                    y_pred = logits.detach().cpu().numpy()
                labels = inputs["labels"]
                labels = labels.detach().cpu().numpy()

                y_l += list(labels)
                y_p += list(y_pred)
            
            acc = sum(np.array(y_l) == np.array(y_p)) / len(y_l)

            if self.task == "STS-B":
                y_p = [float(i) for i in y_p]
                pearson_corr = pearsonr(y_p, y_l)[0] 
                spearman_corr = spearmanr(y_p, y_l)[0]
                
                logger('Eval Epoch: {} [{}/{}] Pearson: {:.4f} Spearman: {:.4f}'
                        .format(epoch, epoch, numepoch, pearson_corr, spearman_corr))

                # logging.debug(f'------pearson_corr:{pearson_corr}, spearman_corr:{spearman_corr}------')
                return pearson_corr

            elif self.task == "CoLA":
                mcc = matthews_corrcoef(y_l, y_p)
                logger('Eval Epoch: {} [{}/{}] Matthews: {:.4f}'
                        .format(epoch, epoch, numepoch, mcc)) 
                # logging.debug(f'------------Acc:{acc}, Mcc:{mcc}------------')
                return mcc

            else:
                f1 = f1_score(y_l, y_p) if self.task != "MNLI" else f1_score(y_l, y_p, average="macro")
                # logging.debug(f'------------Acc:{acc}, F1:{f1}------------')
                logger('Eval Epoch: {} [{}/{}] Acc: {:.4f} F1 score: {:.4f}'
                       .format(epoch, epoch, numepoch, acc, f1))
                return acc       