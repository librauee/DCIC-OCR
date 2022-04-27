import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os


train_dir = '/data/raw_data/train'
files = os.listdir(train_dir)
img_paths = []
for img in files:
    img_path = os.path.join(train_dir, img)
    img_paths.append(img_path)

print(img_paths[:5])

data = pd.DataFrame({'image_path': img_paths})

data['label'] = data['image_path'].apply(lambda x: x.split('.')[0][-4:])
# print(data.info())
# print(data['label'].str.split("", expand=True))
# data[[f'label_{i}' for i in range(1, 5)]] = data['label'].str.split("", expand=True)
for i in range(1, 5):
    data[f'label_{i}'] = data['label'].apply(lambda x: x[i - 1])

print(data.head())

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=1009)
data["kfold"] = -1

for fold, (trn_, val_) in enumerate(mskf.split(data, data[[f'label_{i}' for i in range(1, 5)]])):
    data.loc[val_, "kfold"] = fold

print(data['kfold'].value_counts())


data.to_csv('/data/user_data/train_folds.csv', index=False)


# def split_data(fold):
#     """
#     :param files:
#     :return:
#     """
#     data = pd.read_csv('train_folds.csv')
#     train_fold = data[data['kfold'] != fold]
#     val_fold = data[data['kfold'] == fold]
#     return train_fold.image_path.values, val_fold.image_path.values
#
#
# a, b = split_data(fold=0)
# print(len(a))
# print(a[:5])






