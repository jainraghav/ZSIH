"""
Sketch Data Loader
"""

import torch
from torch.utils.data.dataset import Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import os
import pdb
from PIL import Image
import pandas as pd
from numpy import array

class SketchDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):

        tmp_df = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform

        self.X_train = tmp_df['ImagePath']

        arr = tmp_df['ImagePath'].str.partition('/')[0].values.tolist()
        values = array(arr)

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded).astype(float)
        self.y_train = integer_encoded

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index])
        #img.show()
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)
