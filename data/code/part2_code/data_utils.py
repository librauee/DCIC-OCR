
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import GroupKFold, KFold
import torch 
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import warnings 
import cv2
from albumentations.pytorch.transforms import ToTensorV2
warnings.filterwarnings('ignore')

import albumentations as alb

import random 

# 将数据集划分训练集和验证集
def split_data(files):
    """
    :param files:
    :return:
    """
    random.shuffle(files)
    # 计算比例系数，分割数据训练集和验证集
    ratio = 0.9
    offset = int(len(files) * ratio)
    train_data = files[:offset]
    val_data = files[offset:]
    return train_data, val_data


# 对数据集进行随机打乱
def random_data(files):
    # 设置随机种子，保证每次随机值都一致
    random.seed(2022)
    random.shuffle(files)
    return files

from PIL import Image
import torch

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)


def img_loader(img_path):
#     img = Image.open(img_path)
    
#     return img.convert('RGB')
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def make_dataset(data_path, alphabet, num_class, num_char):
    samples = []
    for img_path in data_path:
        target_str = img_path.split('.png')[0][-4:]
        assert len(target_str) == num_char
        target = []
        for char in target_str:
            vec = [0] * num_class
            vec[alphabet.find(char)] = 1
            target += vec
        samples.append((img_path, target))
    print(len(samples))
    return samples

def generate_data():
    from pathlib import Path 
    import pandas as pd 
    train_list = map(str,sorted(Path('/data/raw_data/train').glob('*.png')))
    train = pd.DataFrame({
        'img_path':train_list,
    })
    test_list = map(str,sorted(Path('/data/raw_data/test').glob('*.png')))
    test = pd.DataFrame({
        'img_path':test_list,
    })
    train['label'] = train['img_path'].map(lambda x:x.split('/')[-1].replace('.png','').replace('_',''))
    print(train.head())

    # train = pd.read_csv('data/train.csv')
    features = 'img_path'
    label = 'label'
    # gkf = GroupKFold(n_splits = 5)
    gkf = KFold(n_splits = 5,shuffle=True)
    for i, (train_index, test_index) in enumerate(gkf.split(train[features], train[label])):
        train.loc[test_index,'fold'] = i 
    train['fold'] = train['fold'].astype(int)
    train.to_csv('/data/user_data/train.csv',index=False,)
    print('data saved in ./data')
    test.to_csv('/data/user_data/test.csv',index=False)
# 验证数据处理类
class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=62, num_char=4,
                 transform=None, target_transform=None, alphabet=alphabet,is_test=False):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.is_test= is_test
        self.samples = make_dataset(self.data_path, self.alphabet,
                                    self.num_class, self.num_char)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.is_test:
            return img
        else:
            return img, torch.Tensor(target)


def load_transform(train_mean,train_std,data_config):
    train_transform = transforms.Compose([
        transforms.Resize(data_config['img_size']),  # 图像放缩
        transforms.RandomRotation((-5, 5)),  # 随机旋转
        # transforms.RandomVerticalFlip(p=0.2),  # 翻转
        # transforms.RandomAffine(0, None, None, (0, 45)), ## 扭曲
        transforms.ToTensor(),  # 转化成张量
        transforms.Normalize(
            mean=train_mean,
            std=train_std
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize(data_config['img_size']),  # 图像放缩
        transforms.ToTensor(),  # 转化成张量
        transforms.Normalize(
            mean=train_mean,
            std=train_std
        )
    ])

    return train_transform,val_transform
