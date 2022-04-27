import os
import json
import torch
import argparse
import albumentations as alb
import cv2
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from captcha_dataset import CaptchaData
from torch.utils.data import DataLoader
import config
import time
import random
import pandas as pd
import numpy as np
from get_norm import init_normalize
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR,
                                      ReduceLROnPlateau
                                      )

from fmix import sample_mask, make_low_freq_image, binarise_mask
from torch.cuda.amp import autocast, GradScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1009, required=False)
    parser.add_argument("--fold", type=int, default=0, required=False)
    parser.add_argument("--h", type=int, default=400, required=False)
    parser.add_argument("--w", type=int, default=1000, required=False)
    parser.add_argument("--num_classes", type=int, default=248, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--num_epoch", type=int, default=124, required=False)
    parser.add_argument("--lr", type=int, default=5e-4, required=False)
    parser.add_argument("--checkpoints", type=str, default='v7', required=False)
    parser.add_argument("--train_dir", type=str, default='/data/raw_data/train', required=False)
    parser.add_argument("--test_dir", type=str, default='/data/raw_data/test', required=False)
    return parser.parse_args()

def split_data(fold, use_extra=True):
    data = pd.read_csv('/data/user_data/train_folds.csv')
    train_fold = data[data['kfold'] != fold]
    val_fold = data[data['kfold'] == fold]
    if use_extra:
        data_extra = pd.read_csv('/data/user_data/mkdata.csv').sample(frac=0.8, random_state=fold)
        return list(train_fold.image_path.values) + list(data_extra.image_path.values), val_fold.image_path.values

    else:
        return train_fold.image_path.values, val_fold.image_path.values

# 计算准确率
def calculat_acc(output, target):
    output, target = output.view(-1, 62), target.view(-1, 62)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc

# 设置随机种子，代码可复现
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def mixup(x, y, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

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

def train(fold, model, loss_func, optimizer, checkpoints, epochs, lr_scheduler=None, ema_model=None):
    print('Train......................')
    # 记录每个epoch的loss和acc
    record = []
    best_acc = 0
    best_epoch = 0
    # 训练过程
    scaler = GradScaler()
    
    for epoch in range(1, epochs):
        # 设置计时器，计算每个epoch的用时
        start_time = time.time()
        model.train()  # 保证每一个batch都能进入model.train()的模式
        # 记录每个epoch的loss和acc
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        for i, (inputs, labels) in enumerate(train_data):
            # print(i, inputs, labels)

            with autocast():
                inputs = inputs.to(device)
                labels = labels.to(device)
                enhance_prob = torch.rand(1)[0]
                if enhance_prob < 0.2:
                    lam = np.clip(np.random.beta(1, 1), 0.6, 0.7)
                    # Make mask, get mean / std
                    mask = make_low_freq_image(3, (args.h, args.w))
                    mask = binarise_mask(mask, lam, (args.h, args.w), True)

                    mask_torch = torch.from_numpy(mask).cuda()
                    mask_torch = mask_torch.type(torch.cuda.FloatTensor)
                    rand_index = torch.randperm(inputs.size()[0]).cuda()
                    inputs = mask_torch * inputs + (1. - mask_torch) * inputs[rand_index, :]

                    rate = mask.sum() / args.w / args.h
                    target = rate * labels + (1. - rate) * labels[rand_index]

                    outputs = model(inputs)
                    loss = loss_func(outputs, target)

                elif enhance_prob < 0.4:
                    lam = np.random.beta(1, 1)
                    rand_index = torch.randperm(inputs.size()[0]).cuda()
                    target_a = labels
                    target_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    # compute output
                    outputs = model(inputs)
                    loss = loss_func(outputs, target_a) * lam + loss_func(outputs, target_b) * (1. - lam)
                else:
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                optimizer.zero_grad()
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # 计算准确率
                acc = calculat_acc(outputs, labels)
                train_acc.append(float(acc))
                train_loss.append(float(loss))
                if epoch > 100:
                    ema_model.update(model)
        if lr_scheduler:
            lr_scheduler.step()
        # 验证集进行验证
        with torch.no_grad():
            model.eval()
            for i, (inputs, labels) in enumerate(val_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 预测输出
                outputs = model(inputs)
                # 计算损失
                loss = loss_func(outputs, labels)
                # 计算准确率
                acc = calculat_acc(outputs, labels)
                val_acc.append(float(acc))
                val_loss.append(float(loss))

        # 计算每个epoch的训练损失和精度
        train_loss_epoch = torch.mean(torch.Tensor(train_loss))
        train_acc_epoch = torch.mean(torch.Tensor(train_acc))
        # 计算每个epoch的验证集损失和精度
        val_loss_epoch = torch.mean(torch.Tensor(val_loss))
        val_acc_epoch = torch.mean(torch.Tensor(val_acc))
        # 记录训练过程
        record.append(
            [epoch, train_loss_epoch.item(), train_acc_epoch.item(), val_loss_epoch.item(), val_acc_epoch.item()])
        end_time = time.time()
        print(
            'epoch:{} | time:{:.4f} | train_loss:{:.4f} | train_acc:{:.4f} | eval_loss:{:.4f} | val_acc:{:.4f}'.format(
                epoch,
                end_time - start_time,
                train_loss_epoch,
                train_acc_epoch,
                val_loss_epoch,
                val_acc_epoch))

        # 记录验证集上准确率最高的模型
        best_model_path = checkpoints + "/" f'best_model_{fold}_b4.pth'
        if val_acc_epoch >= best_acc:
            best_acc = val_acc_epoch
            best_epoch = epoch
            torch.save(model, best_model_path)
        print('Best Accuracy for Validation :{:.4f} at epoch {:d}'.format(best_acc, best_epoch))
        # 每迭代50次保存一次模型
        # if epoch % 50 == 0:
        #     model_name = '/epoch_' + str(epoch) + '.pt'
        #     torch.save(model, checkpoints + model_name)
    # 保存最后的模型
    # torch.save(model, checkpoints + '/last.pt')
    # 将记录保存下下来
    record_json = json.dumps(record)
    with open(checkpoints + '/' + f'record_{fold}_b4.txt', 'w+', encoding='utf8') as ff:
        ff.write(record_json)

import timm
class OCRClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 设置随机种子
    seed_it(args.seed)
    # 分类类别数
    num_classes = args.num_classes
    # batchsize大小
    batch_size = args.batch_size
    # 迭代次数epoch
    epochs = args.num_epoch
    # 学习率
    lr = args.lr
    # 模型保存地址
    checkpoints = '/data/user_data/'+args.checkpoints
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    # 训练接和验证集地址
    train_dir = args.train_dir
    test_dir = args.test_dir
    # 计算均值和标准差

    Width = args.w
    Height = args.h
    train_mean, train_std = [0.8408785860737165, 0.8431024999260902, 0.8397315841118494], [0.20006747062181432, 0.19766466962620616, 0.20183314211890102]

    train_transform = alb.Compose([
        alb.Resize(Height, Width, p=1),
        alb.OneOf([
            alb.OpticalDistortion(distort_limit=0.02, shift_limit=0.02, p=0.2),
            alb.GridDistortion(distort_limit=0.1, p=0.2),
            alb.IAAPiecewiseAffine(scale=(0.01, 0.02), p=0.2),
        ], p=0.1),
        alb.ShiftScaleRotate(rotate_limit=5, shift_limit=0.0625, scale_limit=0.1, p=0.2,
                             border_mode=cv2.BORDER_REPLICATE),
        alb.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2),
        alb.Normalize(
            mean=train_mean,
            std=train_std
        )
    ],

    )
    val_transform = alb.Compose([
        alb.Resize(Height, Width, p=1),
        alb.Normalize(
            mean=train_mean,
            std=train_std
        )
    ])


    train_paths, val_paths = split_data(args.fold, True)
    # 加载训练数据集，转化成标准格式
    train_dataset = CaptchaData(train_paths, transform=train_transform)
    # 加载数据
    train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=16
                            )
    # 加载验证集，转化成标准格式
    val_dataset = CaptchaData(val_paths, transform=val_transform)
    val_data = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=16
                          )
    print('训练集数量：%s   验证集数量：%s' % (train_dataset.__len__(), val_dataset.__len__()))


    # model = OCRClassifier(model_arch='tf_efficientnet_b4_ns', n_class=248, pretrained=True)

    ema_decay = 0.0
    # model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=248)
    model = OCRClassifier(model_arch='tf_efficientnet_b4_ns', n_class=248, pretrained=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ema_model = timm.utils.ModelEmaV2(model, ema_decay, device)

    loss_func = nn.MultiLabelSoftMarginLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=5e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    train(args.fold, model, loss_func, optimizer, checkpoints, epochs, scheduler, ema_model)
