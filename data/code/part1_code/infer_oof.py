import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from get_norm import init_normalize
import csv
from tqdm import tqdm
import config
import albumentations as alb
import cv2
import numpy as np
import pandas as pd
from captcha_dataset import CaptchaData_valid
from torch.utils.data import DataLoader
import timm
import argparse



class OCRClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

def img_loader(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def split_data(fold, use_extra=True):
    data = pd.read_csv('/data/user_data/train_folds.csv')
    train_fold = data[data['kfold'] != fold]
    val_fold = data[data['kfold'] == fold]
    if use_extra:
        data_extra = pd.read_csv('/data/user_data/mkdata.csv').sample(frac=0.8, random_state=fold)
        return list(train_fold.image_path.values) + list(data_extra.image_path.values), val_fold.image_path.values

    else:
        return train_fold.image_path.values, val_fold.image_path.values

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=400, required=False)
    parser.add_argument("--w", type=int, default=1000, required=False)
    parser.add_argument("--use_model", type=str, default='v4', required=False)
    parser.add_argument("--use_softmax", type=str, default='', required=False)
    return parser.parse_args()


args = parse_args()
Height = args.h
Width = args.w
no_softmax = args.use_softmax        # '_no_softmax'
use_model = args.use_model

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)

if Height == 280:
    test_mean, test_std = [0.8408704800764719, 0.8430948053002357, 0.839723514576753], [
        0.20036302471856277, 0.19795909674788514, 0.20213062292287748]
    # 280 700
elif Height == 360:
    test_mean, test_std = [0.8408691800077757, 0.843093117582798, 0.8397223384737968], [
        0.20037801617731651, 0.19797363897711037, 0.20214536744058131]
    # 360 900
elif Height == 400:
    test_mean, test_std = [0.8408785860737165, 0.8431024999260902, 0.8397315841118494], [
        0.20006747062181432, 0.19766466962620616, 0.20183314211890102]

val_transform = alb.Compose([
    alb.Resize(Height, Width, p=1),
    alb.Normalize(
        mean=test_mean,
        std=test_std
    )
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images_path_total = []
res = []
for fold in range(5):
    _, images = split_data(fold=fold, use_extra=False)
    images_path_total.extend(images)

    val_dataset = CaptchaData_valid(images, transform=val_transform)
    val_data = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                          #num_workers=16
                          
                          )
    print('测试集数量：%s' % (val_dataset.__len__()))

    model = torch.load(f'/data/user_data/{use_model}/best_model_{fold}_b4.pth').cuda()
    with torch.no_grad():
        model.eval()
        for i, inputs in tqdm(enumerate(val_data)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            if len(no_softmax) == 0:
                outputs = outputs.view(-1, 62)
                outputs = nn.functional.softmax(outputs, dim=1)
                outputs = outputs.view(-1, 248)

            outputs = outputs.cpu().numpy()
            res.append(outputs)

total = np.concatenate(res)

pred = pd.DataFrame(total)
pred.columns = [f'prob_{i}' for i in range(248)]
pred['image_path'] = images_path_total
print(pred.tail())



if not os.path.exists('/data/user_data/infer/'):
    os.makedirs('/data/user_data/infer/')
pred.to_pickle(f'/data/user_data/infer/oof_{use_model}{no_softmax}.pkl')


