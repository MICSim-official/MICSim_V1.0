import configparser
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from Accuracy.src.utils import  misc,make_path
from Accuracy.src.Modules.CNN import optimizer
import os
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler

logger = misc.logger.info



config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
num_epoch          = int(config['Training']['numEpoch'])
decreasing_lr      =     config['Training']['decreasing_lr']
quantization_mode  =     config['Quantization']['mode']
train_log_interval = int(config['Training']['train_log_interval'])
val_log_interval   = int(config['Training']['val_log_interval'])

class Trainer:
    def __init__(self,model,train_loader,test_loader):
        self.model = model
        self.decreasing_lr = list(map(int, decreasing_lr.split(',')))
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer.optimizer(self.model)
        self.criterion = optimizer.loss_func()
        self.cuda = torch.cuda.is_available()
        self.best_acc = 0
        self.old_file = None
        self.mode = None
        self.logdir = make_path.makepath_logdir()
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
    
    def Training_with_Timer(self):
        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        logger('start time: {}'.format(current_time))
        self.optimize()
        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        logger('finish time: {}'.format(current_time))
    
    def Training(self):
        self.optimize()
  
    def _update_one_batch(self, data, target):
        self.model.train()
        # forward calculation
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        # backward calculation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_one_epoch(self,epoch):
        running_loss = 0.0
        running_correct=0
        total = 0
        total_samples = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            indx_target = target.clone()
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self._update_one_batch(data, target)
            #assert 0 > 1
            # calculate training loss and accuracy
            if batch_idx % train_log_interval == 0 and batch_idx >= 0:
                self.model.eval()
                output = self.model(data)
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) * 1.0 / len(data)
                running_correct += correct
                loss = self.criterion(output, target)
                running_loss += loss.data.cpu().item() * data.size(0)
                total += target.size(0)
                total_samples += data.size(0)
                # self.train_losses.append(loss.data.cpu().item())
                # self.train_accuracies.append(acc)
                logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    loss.data, acc, self.optimizer.param_groups[0]['lr']))
        self.scheduler.step()
        epoch_loss = running_loss / total_samples
        epoch_acc = 100. * float(running_correct) / total
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)

    def optimize(self):
        for epoch in range(num_epoch):
            self._update_one_epoch(epoch)

            if epoch in self.decreasing_lr:
                if quantization_mode =='WAGE':
                    self.optimizer.param_groups[0]['lr'] *= 0.125
                    #self.optimizer.bn_lr *= 0.25
                else:
                    self.optimizer.param_groups[0]['lr'] *= 0.1
            if epoch % val_log_interval == 0:
                self._val(epoch)

    def one_epoch_finetuning(self):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx == int(0.5*len(self.train_loader)) or batch_idx == int(0.8*len(self.train_loader)) :
                if quantization_mode =='WAGE':
                    self.optimizer.param_groups[0]['lr'] *= 0.125
                    #self.optimizer.bn_lr *= 0.25
                else:
                    self.optimizer.param_groups[0]['lr'] *= 0.1

            indx_target = target.clone()
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self._update_one_batch(data, target)
            #assert 0 > 1
            # calculate training loss and accuracy
            if batch_idx % train_log_interval == 0 and batch_idx >= 0:
                self.model.eval()
                output = self.model(data)
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) * 1.0 / len(data)
                loss = self.criterion(output, target)
                logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    0, batch_idx * len(data), len(self.train_loader.dataset),
                    loss.data, acc, self.optimizer.param_groups[0]['lr']))
    def _val(self,epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        logger("===================== testing phase =====================")
        for i, (data, target) in enumerate(self.test_loader):
            # calculate test loss and accuracy for batch
            indx_target = target.clone()
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                test_loss_i = self.criterion(output, target)
                test_loss += test_loss_i.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.cpu().eq(indx_target).sum()

        # calculate total test loss and accuracy
        test_loss = test_loss / len(self.test_loader)  # average over number of mini-batch
        acc = 100. * correct / len(self.test_loader.dataset)
        self.test_losses.append(test_loss.data.cpu().item())
        self.test_accuracies.append(acc)
        logger('\tEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch, test_loss, correct, len(self.test_loader.dataset), acc))

        # save model if the test accurcy is higher than the past best
        if acc > self.best_acc:
            new_file = os.path.join(self.logdir, 'best-{}.pth'.format(epoch))
            misc.model_save(self.model, new_file, old_file=self.old_file, verbose=True)
            self.best_acc = acc
            self.old_file = new_file