import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import config

# 数据标签
source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)

# 读取图像，并转化格式
def img_loader(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 制作数据集
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



from torchvision.transforms.functional import to_tensor
# 验证数据处理类
class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=62, num_char=4,
                 transform=None, target_transform=None, alphabet=alphabet):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = make_dataset(self.data_path, self.alphabet,
                                    self.num_class, self.num_char)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:

            img = self.transform(image=img)['image']
            # print(img)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = torch.tensor(img, dtype=torch.float)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, torch.Tensor(target)

class CaptchaData_test(Dataset):
    def __init__(self, data_path, num_class=62, num_char=4,
                 transform=None, target_transform=None, alphabet=alphabet):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = data_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img_path = (os.path.join(config.test_dir, img_path))
        img = img_loader(img_path)
        if self.transform is not None:

            img = self.transform(image=img)['image']
            # print(img)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = torch.tensor(img, dtype=torch.float)


        return img


class CaptchaData_valid(Dataset):
    def __init__(self, data_path, num_class=62, num_char=4,
                 transform=None, target_transform=None, alphabet=alphabet):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = data_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        # img_path = (os.path.join(config.test_dir, img_path))
        img = img_loader(img_path)
        if self.transform is not None:

            img = self.transform(image=img)['image']
            # print(img)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = torch.tensor(img, dtype=torch.float)


        return img
