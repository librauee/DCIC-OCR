# %%
from pathlib import Path
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
warnings.filterwarnings('ignore')

import os

# train = train[:100000]
# train = train[train['time_id']>=700].reset_index(drop=True)
# train.head()
# train['time_id'].unique()

# %%
import pytorch_lightning as pl 
import torch.nn.functional as F 
from torch import nn
import gc,os
import torch 
from pytorch_lightning import LightningDataModule,LightningModule
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import GroupKFold
from my_trainer import model_config

import torch
from data_utils import CaptchaData, generate_data, load_transform
from my_trainer import LitModel

from utils import CSVLogger, FlexibleTqdm, LearningCurve, ModelCheckpoint, RegressionMetric

import cv2 as cv
import cv2
import os
import numpy as np
import torch
import torchvision


# 计算数据集的标准差和均值
def init_normalize(imgs_path_list, size):
    img_h, img_w = size[0], size[1]  # 根据自己数据集适当调整，影响不大
    means = [0, 0, 0]
    stdevs = [0, 0, 0]
    img_list = []
    
    # imgs_path_list = os.listdir(data_dir)

    num_imgs = 0
    # print(data)
    for pic in imgs_path_list:
        # print(pic)
        num_imgs += 1
        img = cv.imread(pic)
        img = cv.resize(img, (img_h, img_w))
        img = img.astype(np.float32) / 255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()

    means.reverse()
    stdevs.reverse()
    means = np.asarray(means) / num_imgs
    stdevs = np.asarray(stdevs) / num_imgs
    # print("normMean = {}".format(means))
    # print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
    print(list(means), list(stdevs))
    return list(means), list(stdevs)



from models import *



trainer_config={
  'gpus': 1,
  'accumulate_grad_batches': 1,
  'max_epochs':70,
#   'progress_bar_refresh_rate': 1,
 'precision':16,
  'fast_dev_run': False,
  'num_sanity_val_steps': 0,
  'resume_from_checkpoint': None,
}

data_config = {
            'img_size':model_config['img_size'],
          'train_loader':{
              'batch_size': 16,
              'shuffle': True,
              'num_workers': 8,
              'pin_memory': False,
              'drop_last': True,
          },
          'val_loader': {
              'batch_size': 16,
              'shuffle': False,
              'num_workers': 8,
              'pin_memory': False,
              'drop_last': False
         },}

def run_one_fold(fold):
    train = pd.read_csv('/data/user_data/train.csv')
    print(train.head())
    mkdata = pd.read_csv('/data/user_data/mkdata.csv').sample(2000,replace=False,random_state=fold).reset_index(drop=True)
    train = pd.concat([train,mkdata],ignore_index=True,axis=0)
    train_data = train[train['fold']!=fold].reset_index(drop=True)
    # train_data = train.reset_index(drop=True)
    val_data = train[train['fold']==fold].reset_index(drop=True)
    train_mean, train_std = init_normalize(train_data['img_path'],size=data_config['img_size'])
    train_transform,val_transform = load_transform(train_mean,train_std,data_config)

    train_dataset = CaptchaData(train_data['img_path'], transform=train_transform)
    train_data = DataLoader(dataset=train_dataset, **data_config['train_loader'])
    # 加载验证集，转化成标准格式
    val_dataset = CaptchaData(val_data['img_path'], transform=val_transform)
    val_data = DataLoader(dataset=val_dataset, **data_config['val_loader'])
    print('训练集数量: %s   验证集数量: %s' % (train_dataset.__len__(), val_dataset.__len__()))

    model = LitModel(model_name='b5',model_config=model_config)
    ft = FlexibleTqdm(train_dataset.__len__()//data_config['train_loader']['batch_size'], column_width=12)
    csvlog = CSVLogger(dirpath=f'/data/user_data/history/{model_config["version"]}/',filename=f'fold_{fold}')
    lc  = LearningCurve(figsize=(12, 4), names=("loss", "acc",),file_path=f'/data/user_data/history/{model_config["version"]}/')
    loss_checkpoint = ModelCheckpoint(dirpath=f'/data/user_data/ckp/{model_config["version"]}/',patience=100,
        filename=f'fold_{fold}',monitor='val_acc',mode='max')

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        callbacks=[loss_checkpoint,ft,csvlog,lc],
        **trainer_config,
    )

    trainer.fit(model, train_dataloader=train_data,
                val_dataloaders=val_data,
               )
    del trainer
    del model
    gc.collect()

# %%
if __name__ == '__main__':
    generate_data()
    for f in range(5):
        print(f'====training fold {f}====')
        rn = run_one_fold(f)
        del rn
        gc.collect()
