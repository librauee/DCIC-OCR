
import numpy as np
from pytorch_lightning import LightningModule
from torch import nn
import torch 

from utils import ClassificationMetric, MultiLabelMetric, RegressionMetric, cutmix, fmix, mixup 
from models import *
from efficientnet_pytorch import EfficientNet
import timm 
import warnings 
warnings.filterwarnings('ignore')
model_list = {
'b1':EfficientNet.from_pretrained,
'b5':EfficientNet.from_pretrained,
'b7':EfficientNet.from_pretrained,
'swin_large':SwinForClass_1k,
'swin_base':SwinForClass_1k,
'resnext50_32x4d':SwinForClass_1k,
'b5_train':SwinForClass_1k,
'b7_train':SwinForClass_1k,
'b4_train':SwinForClass_1k,
'b1_train':SwinForClass_1k,
'densenet':SwinForClass_1k,
}

## b1+cutmix+mixup+128*256 分数和最高分差不多
model_config = {
    'b1':{'model_name':'efficientnet-b1','num_classes':248,'weights_path':None},
    'b5':{'model_name':'efficientnet-b5','num_classes':248,'weights_path':None},
    'b7':{'model_name':'efficientnet-b7','num_classes':248,'weights_path':None},

    'version':'cutmix_fmix_400_1000_b5_nodata2',
    'img_size':[400,1000],
    'cutmix':True,
    'fmix':True,
    'mixup':False,
}



class LitModel(LightningModule):
    def __init__(self,model_name,model_config):
        super().__init__()

        # self.loss_f = nn.MSELoss()
        self.model_config = model_config
        self.loss_f = nn.MultiLabelSoftMarginLoss()
#         self.loss_f = nn.L1Loss()
        # self.model = SimpleMLP()
        self.model = self.load_model(model_name)
        self.train_metric = MultiLabelMetric(metrics=['acc'])
        self.val_metric = MultiLabelMetric(metrics=['acc'])
        
        self.history = {
            'loss':[],'acc':[],
            'val_loss':[],'val_acc':[],
        }
        

    def load_model(self,model_name):
        return model_list[model_name](**model_config[model_name])

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        mix_decision = np.random.rand()
        if mix_decision <=0.4 and self.model_config['cutmix']:
            x,y = cutmix(x,y,1.)
        elif mix_decision > 0.4 and mix_decision <=0.5 and self.model_config['fmix']:
            x, y = fmix(x, y, alpha=1., decay_power=5., shape=(self.model_config['img_size'][0],self.model_config['img_size'][1]),device=self.device)
            # x,y = mixup(x,y,alpha=0.5)
            x = x.float()
        y_hat = self(x)
        if mix_decision <= 0.5 and (self.model_config['cutmix'] or self.model_config['fmix']):
            loss = self.loss_f(y_hat, y[0]) * y[2] + self.loss_f(y_hat, y[1]) * (1. - y[2])
            self.train_metric.update(y_hat,y[0])
        else:
            loss = self.loss_f(y_hat, y)
        # loss = self.common_step(batch,self.train_metric)
            self.train_metric.update(y_hat,y)

        return loss

    def common_step(self,batch,metric):
        x,y = batch
        y_hat = self(x)
        loss = self.loss_f(y_hat, y)
        metric.update(y_hat,y)
        return loss

    def training_epoch_end(self, outs):
        # 计算平均loss
        loss = 0.
        for out in outs:
            loss += out["loss"].cpu().detach().item()
        loss /= len(outs)
        # 计算指标
        acc = self.train_metric.compute()
        # 记录log
        self.history["loss"].append(loss)
        self.history["acc"].append(acc[0])


    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        # val_loss = self.common_step(batch,self.val_metric)
        val_loss = self.loss_f(y_hat, y)
        self.val_metric.update(y_hat,y)
        return val_loss 

    def validation_epoch_end(self,outs):
        val_loss = sum(outs).item() / len(outs)
        val_acc = self.val_metric.compute()

        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc[0])


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,)
        # optimizer = torch.optim.RAdam(self.parameters(),lr=2e-3,)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=2e-3,)
#         lrs = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2,)
        lrs = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[40], gamma=0.1)
        # lrs = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.001,max_lr=0.05,cycle_momentum=False)

        return [optimizer],[lrs]