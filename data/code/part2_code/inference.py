# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings

from prometheus_client import Counter 
warnings.filterwarnings('ignore')
import os


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

import torch
from data_utils import CaptchaData, generate_data, load_transform
from my_trainer import LitModel

from utils import FlexibleTqdm, ModelCheckpoint, RegressionMetric
from data_utils import alphabet
import cv2 as cv
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

# %%




from models import *
from my_trainer import model_config

# %%

# %%

from baseline import  trainer_config ,data_config


def run_one_fold(fold):
    train = pd.read_csv('/data/user_data/train.csv')
    test = pd.read_csv('/data/user_data/test.csv')
    train_data = train[train['fold']!=fold].reset_index(drop=True)
    val_data = train[train['fold']==fold].reset_index(drop=True)
    train_mean, train_std = init_normalize(train_data['img_path'],size=data_config['img_size'])
    train_transform,val_transform = load_transform(train_mean,train_std,data_config)
    train_dataset = CaptchaData(train_data['img_path'], transform=train_transform)
    # 加载数据
    train_data = DataLoader(dataset=train_dataset, **data_config['train_loader'])
    # 加载验证集，转化成标准格式
    val_dataset = CaptchaData(val_data['img_path'], transform=val_transform,is_test=True)
    val_data = DataLoader(dataset=val_dataset, **data_config['val_loader'])
    test_dataset = CaptchaData(test['img_path'], transform=val_transform,is_test=True)
    test_data = DataLoader(dataset=test_dataset, **data_config['val_loader'])


    print('训练集数量: %s   验证集数量: %s' % (train_dataset.__len__(), val_dataset.__len__()))
    from my_trainer import model_config
    model = LitModel(model_name='b5',model_config=model_config)
    ft = FlexibleTqdm(test_dataset.__len__()//data_config['val_loader']['batch_size'], column_width=12)
    
    loss_checkpoint = ModelCheckpoint(dirpath=f'/data/user_data/ckp/{model_config["version"]}/',patience=5,
        filename=f'fold_{fold}',monitor='val_acc',mode='max')

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        callbacks=[loss_checkpoint,ft],
        **trainer_config,
    )

    prediction = trainer.predict(model,dataloaders=test_data,ckpt_path=f'/data/user_data/ckp/{model_config["version"]}/fold_{fold}.ckpt')
    torch.save(prediction,f'/data/user_data/ckp/{model_config["version"]}/fold_{fold}.pt')
    prediction = trainer.predict(model,dataloaders=val_data,ckpt_path=f'ckp/{model_config["version"]}/fold_{fold}.ckpt')
    torch.save(prediction,f'/data/user_data/ckp/{model_config["version"]}/fold_oof_{fold}.pt')
    del trainer
    del model
    del prediction
    gc.collect()

def pred2char(preds):
    final_predictions = []
    for p in preds:
        output = p.view(-1, 62)
        output = nn.functional.softmax(output.float(), dim=1)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, 4).cpu().numpy().tolist()
        f = lambda x : ''.join([alphabet[i] for i in x])
        final_predictions += list(map(f,output))
    return final_predictions
def count_max(some_dict):
    res = {}
    for i in some_dict:
        if i not in res:
            res[i] = 1
        else:
            res[i] += 1
    # print(res)
    bst = -1
    ret = ''
    for i in res:
        if res[i] > bst:
            bst = res[i]
            ret = i
    return ret 
# %%
if __name__ == '__main__':
    # generate_data()
    predictions = []
#     version1 = model_config['version']
    
    version1 = 'cutmix_fmix_400_1000_b5_nodata2'
    # model_config['version'] = version1
    # version2 = 'cutmix_fmix_b5_5fold'
    # torch.save()
    for f in range(5):
        print(f'====training fold {f}====')
        rn = run_one_fold(f)

    import pandas as pd 
    train = pd.read_csv('/data/user_data/train.csv')
    for i in range(248):
        train['prob_'+str(i)] = 0
    pro_f = ['prob_'+str(i) for i in range(248)]

    for i in range(5):
        pt_path = '/data/user_data/ckp/cutmix_fmix_400_1000_b5_nodata2/'
        p = torch.cat(torch.load(pt_path+f'fold_oof_{i}.pt'),dim=0)
        output = p.view(-1, 62)
        output = nn.functional.softmax(output.float(), dim=1)
        output = output.view(-1, 248).cpu().numpy()
        train.loc[train['fold']==i,pro_f] = output
    train.to_pickle('/data/user_data/infer/oof_lh.pkl')

    predictions = []
    for i in range(5):
        pt_path = '/data/user_data/ckp/cutmix_fmix_400_1000_b5_nodata2/'
        p = torch.cat(torch.load(pt_path+f'fold_{i}.pt'),dim=0)
        predictions.append(p)
    predictions = torch.stack(predictions,dim=2).mean(2)
    output = predictions.view(-1, 62)
    output = nn.functional.softmax(output.float(), dim=1)
    output = output.view(-1, 248).cpu().numpy()
    test = pd.read_csv('/data/user_data/test.csv')
    pro_f = ['prob_'+str(i) for i in range(248)]
    test['num'] = test['img_path'].map(lambda x:x.split('/')[2].replace('.png','')).astype(int)
    test[pro_f] = output
    test.sort_values('num',ascending=True,inplace=True,ignore_index=True)
    test[pro_f].to_csv('/data/user_data/infer/predict_lh.csv',index=False)

