
from sklearn.metrics import *
import gc ,os
import numpy as np 
import pandas as pd 
from pytorch_lightning.callbacks import Callback
import time 
import torch
import warnings 
warnings.filterwarnings('ignore')

from fmix import sample_mask 


class ClassificationMetric(object): # 记录结果并计算指标

    def __init__(self, accuracy=True, recall=True, precision=True, f1=True, average="macro"):
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.f1 = f1
        self.average = average

        self.preds = []
        self.target = []

    def reset(self): # 重置结果
        self.preds.clear()
        self.target.clear()
        gc.collect()

    def update(self, preds, target): # 更新结果
        preds = torch.sigmoid(preds)
        preds = list(preds.cpu().detach().numpy()>0.5)
        target = list(target.cpu().detach().numpy()>0.5) if target.dim() > 1 else list(target.cpu().detach().numpy()>0.5)
        self.preds += preds
        self.target += target

    def compute(self): # 计算结果
        metrics = []
        if self.accuracy:
            metrics.append(accuracy_score(self.target, self.preds))
        if self.recall:
            metrics.append(recall_score(self.target, self.preds,average=self.average))
        if self.precision:
            metrics.append(precision_score(self.target, self.preds, average=self.average))
        if self.f1:
            metrics.append(f1_score(self.target, self.preds, average=self.average))
        self.reset()
        return metrics

from scipy.special import softmax
class MultiLabelMetric(object): # 记录结果并计算指标

    def __init__(self, metrics:list):
        self.metrics = metrics

        self.preds = []
        self.target = []

    def reset(self): # 重置结果
        self.preds.clear()
        self.target.clear()
        gc.collect()

    def update(self, preds, target): # 更新结果
        # preds = torch.sigmoid(preds)
        preds = list(preds.cpu().detach().numpy())
        target = list(target.cpu().detach().numpy()) 
        self.preds += preds
        self.target += target
    
    def calculat_acc(self,output, target):
        output = np.array(output)
        target = np.array(target)
        output, target = output.reshape((-1,4,62)), target.reshape((-1,4,62))
        output = softmax(output, axis=2)
        output = np.argmax(output, axis=2)
        target = np.argmax(target, axis=2)
        # output, target = output.reshape(-1, 4), target.reshape(-1, 4)
        correct_list = []
        for i, j in zip(target, output):
            if (i == j).all():
                correct_list.append(1)
            else:
                correct_list.append(0)
        acc = sum(correct_list) / len(correct_list)
        return acc

    def compute(self): # 计算结果
        metrics = []
        for m in self.metrics:
            if m == 'acc':
                metrics.append(self.calculat_acc(self.preds, self.target))

        self.reset()
        return metrics


class CSVLogger(Callback):
    def __init__(self, dirpath="history/", filename="history"):
        super(CSVLogger, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".csv":
            self.name += ".csv"

    def on_train_epoch_end(self, trainer, module): # 在每轮结束时保存log到磁盘
        history = pd.DataFrame(module.history)
        history.to_csv(self.name, index=False)
import matplotlib.pyplot as plt 
class LearningCurve(Callback):
    def __init__(self, figsize=(12, 4),file_path='', names=("loss", "acc", "f1")):
        super(LearningCurve, self).__init__()
        self.figsize = figsize
        self.names = names
        self.file_path = file_path

    def on_fit_end(self, trainer, module):
        history = module.history
        plt.figure(figsize=self.figsize)
        for i, j in enumerate(self.names):
            plt.subplot(1, len(self.names), i + 1)
            plt.title(j + "/val_" + j)
            plt.plot(history[j], "--o", color='r', label=j)
            plt.plot(history["val_" + j], "-*", color='g', label="val_" + j)
            plt.legend()
        plt.savefig(self.file_path+'now.png')


class RegressionMetric(object): # 记录结果并计算指标

    def __init__(self, metrics:list,average="macro"):
        
        # self.extrenal_data = extrenal_data
        self.average = average
        self.metrics = metrics

        self.preds = []
        self.target = []
        self.ts = []

    def reset(self): # 重置结果
        self.preds.clear()
        self.target.clear()
        self.ts.clear()
        gc.collect()

    def update(self, preds, target,ts): # 更新结果
        # preds = torch.sigmoid(preds)
        preds = list(preds.cpu().detach().numpy())
        target = list(target.cpu().detach().numpy())
        ts = list(ts.cpu().detach().numpy())
        
        self.preds += preds
        self.target += target
        self.ts += ts 


    def compute(self): # 计算结果
        metrics = []
        for m in self.metrics:
            if m == 'mse':
                metrics.append(mean_squared_error(self.target, self.preds))
            if m == 'rmse':
                metrics.append(np.sqrt(mean_squared_error(self.target,self.preds)))
            if m == 'mape':
                metrics.append(mean_absolute_percentage_error(self.target, self.preds))
            if m == 'corr':
                nts = np.array(self.ts).reshape(-1)
                y_pred = np.array(self.preds).reshape(-1)
                y_true = np.array(self.target).reshape(-1)
                corr = []
                # print(nts)
                # print(y_pred.shape,y_true.shape)
                for t in set(list(nts)):
                    t_mask = (nts == t)
                    corr.append(np.corrcoef(y_pred[t_mask],y_true[t_mask])[0][1])
                metrics.append(np.mean(corr))
        self.reset()
        return metrics


class FlexibleTqdm(Callback):
    def __init__(self, steps, column_width=10):
        super(FlexibleTqdm, self).__init__()
        self.steps = steps
        self.column_width = column_width
        self.info = "\rEpoch_%d %s%% [%s]"

    def on_train_start(self, trainer, module):
        history = module.history
        self.row = "-" * (self.column_width + 1) * (len(history) + 2) + "-"
        title = "|"
        title += "epoch".center(self.column_width) + "|"
        title += "time".center(self.column_width) + "|"
        for i in history.keys():
            title += i.center(self.column_width) + "|"
        print(self.row)
        print(title)
        print(self.row)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        current_index = int((batch_idx + 1) * 100 / self.steps)
        tqdm = ["."] * 100
        for i in range(current_index - 1):
            tqdm[i] = "="
        if current_index:
            tqdm[current_index - 1] = ">"
        print(self.info % (module.current_epoch, str(current_index).rjust(3), "".join(tqdm)), end="")

    def on_train_epoch_start(self, trainer, module):
        print(self.info % (module.current_epoch, "  0", "." * 100), end="")
        self.begin = time.perf_counter()

    def on_train_epoch_end(self, trainer, module):
        self.end = time.perf_counter()
        history = module.history
        detail = "\r|"
        detail += str(module.current_epoch).center(self.column_width) + "|"
        detail += ("%d" % (self.end - self.begin)).center(self.column_width) + "|"
        for j in history.keys():
            # print(j)
            j = history[j]
            if len(j) == 0:
                j = [0]
            detail += ("%.06f" % j[-1]).center(self.column_width) + "|"
        print("\r" + " " * 120, end="")
        print(detail)
        print(self.row)



class ModelCheckpoint(Callback):
    def __init__(self, dirpath="checkpoint/", filename="checkpoint", 
    patience=3,
    monitor="val_acc", mode="max"):
        super(ModelCheckpoint, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".pth":
            self.name += ".pth"
        self.monitor = monitor
        self.mode = mode
        self.value = -1e6 if mode == "max" else 1e6
        self.pation = 0
        self.mc_patience = patience

    def on_validation_epoch_end(self, trainer, module): # 在每轮结束时检查
        # print(module.history)
        
        if self.mode == "max" and module.history[self.monitor][-1] > self.value:
            # cur_score = module.history[self.monitor][-1]
            self.value = module.history[self.monitor][-1]
            # torch.save(module.state_dict(), self.name)
            trainer.save_checkpoint(self.name.replace('.pth','.ckpt'))
            self.pation = 0
        elif self.mode == 'max':
            self.pation += 1
            # print(module.history[self.monitor])
            if self.pation >= self.mc_patience:
                # print(f'\t stop model because meet max patience {self.patience} ')
                trainer.should_stop = True
        
        if self.mode == "min" and module.history[self.monitor][-1] < self.value:
            self.value = module.history[self.monitor][-1]
            # torch.save(module.state_dict(), self.name)
            trainer.save_checkpoint(self.name.replace('.pth','.ckpt'))
            self.pation = 0
        elif self.mode =='min':
            self.pation += 1
            if self.pation >= self.mc_patience:
                # print('stop model because meet max patience')
                trainer.should_stop = True 


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False,device='cuda'):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    #mask =torch.tensor(mask, device=device).float()
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask).to(device)*data
    x2 = torch.from_numpy(1-mask).to(device)*shuffled_data
    targets=(targets, shuffled_targets, lam)
    return (x1+x2), targets
    
def mixup(x:torch.Tensor, y:torch.Tensor, alpha:float = 1.0):
    """
    Function which performs Mixup augmentation
    """
    assert alpha > 0, "Alpha must be greater than 0"
    assert x.shape[0] > 1, "Need more than 1 sample to apply mixup"

    lam = np.random.beta(alpha, alpha)
    rand_idx = torch.randperm(x.shape[0])
    mixed_x = lam * x + (1 - lam) * x[rand_idx, :]

    target_a, target_b = y, y[rand_idx]

    return mixed_x, (target_a, target_b, lam)