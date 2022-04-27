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
from captcha_dataset import CaptchaData_test
from torch.utils.data import DataLoader
import argparse


# 读取图像，并转化格式
def img_loader(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



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
SUBMIT = False

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)

test_dir = '/data/raw_data/test/'

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

test_mean, test_std = init_normalize(test_dir, size=[Height , Width])
# if Height == 280:
#     test_mean, test_std = [0.8448635504603386, 0.8477284370994568, 0.8426760819864273], \
#                        [0.19310236097380518, 0.18947888918563724, 0.19577497842028738]
#     # 280 700
# elif Height == 360:
#     test_mean, test_std = [0.8448636641526223, 0.8477270549583436, 0.8426755827331543],\
#         [0.19311411200240255, 0.18949073773533107, 0.19578715221069753]
#     # 360 900
# elif Height == 400:
#     test_mean, test_std = [0.8448737227702141, 0.8477376716732978, 0.8426855733823776],[
#         0.19282984165981412, 0.18920864015609026, 0.19550152810238303]
#
#     # 400 1000

val_transform = alb.Compose([
    alb.Resize(Height, Width, p=1),
    alb.Normalize(
        mean=test_mean,
        std=test_std
    )
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images = os.listdir(test_dir)
images.sort(key=lambda x: int(x[:-4]))

val_dataset = CaptchaData_test(images, transform=val_transform)
val_data = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                      #num_workers=16
                      )
print('测试集数量：%s' % (val_dataset.__len__()))

model_paths = [f'/data/user_data/{use_model}/best_model_{i}_b4.pth' for i in range(5)]
total = []

for model_path in model_paths:
    model = torch.load(model_path).cuda()
    res = []
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
    res = np.concatenate(res)
    total.append(res)

total = np.mean(np.array(total), axis=0)
pred = pd.DataFrame(total)
pred.columns = [f'prob_{i}' for i in range(248)]
print(pred.tail())
pred.to_pickle(f'/data/user_data/infer/predict_{use_model}{no_softmax}.pkl')

if SUBMIT:

    total = total.reshape(-1, 62)
    total = np.argmax(total, axis=1)
    total = total.reshape(-1, 4)
    res = []
    for o in total:
        o = ''.join([alphabet[i] for i in o])
        res.append(o)

    result = pd.DataFrame({'num':[i for i in range(1, len(images) + 1)], 'tag':res})
    result.to_csv(f'sub_{use_model}_.csv', index=False)

