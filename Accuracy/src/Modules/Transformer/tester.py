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
task = config['Network']['task']
dump_average_value = config['Quantization']['dumpaveragevalue']
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
class Tester:
    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_dataloader = test_dataloader
        self.task = task
        self.acc  = 0.0
                                      
    def _val(self):
        logger("===================== testing phase =====================")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        y_l, y_p = [], []
        with torch.no_grad():
            for step, batch in enumerate(self.test_dataloader):
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
                
                if dump_average_value == 'True':
                    break
            
            acc = 100. * (sum(np.array(y_l) == np.array(y_p)) / len(y_l))
            if self.task == "STS-B":
                y_p = [float(i) for i in y_p]
                pearson_corr = pearsonr(y_p, y_l)[0] 
                spearman_corr = spearmanr(y_p, y_l)[0]
                
                logger('Test result: Pearson: {:.4f} Spearman: {:.4f}'.format(pearson_corr, spearman_corr))

            elif self.task == "CoLA":
                mcc = matthews_corrcoef(y_l, y_p)
                logger('Test result: Acc: {:.4f}% :Matthews: {:.4f}'.format(acc, mcc)) 

            else:
                f1 = f1_score(y_l, y_p) if self.task != "MNLI" else f1_score(y_l, y_p, average="macro")
                logger('Test result: Acc: {:.4f}% F1 score: {:.6f}'.format(acc, f1))