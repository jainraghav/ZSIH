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
from itertools import permutations

# Testing code is commented out
class SketchImageDataset(Dataset):
    def __init__(self, sketch_data, image_data, sketch_path, image_path, transform=None):

        sketch_df = pd.read_csv(sketch_data)
        image_df = pd.read_csv(image_data)
        self.sketch_path = sketch_path
        self.image_path = image_path
        self.transform = transform

        self.X_sketch_train = sketch_df['ImagePath']
        self.X_image_train = image_df['ImagePath']

        sketch_arr = sketch_df['ImagePath'].values.tolist()
        image_arr = image_df['ImagePath'].values.tolist()

        sketch_classes = [x.split('/')[0] for x in sketch_arr]
        image_classes = [x.split('/')[0] for x in image_arr]
        uniq_classes = list(set(sketch_classes))

        print(uniq_classes)
        image_sketch_hash = []
        for x in uniq_classes:
            indices_sketch = [i for i, e in enumerate(sketch_classes) if e == x]
            indices_image = [i for i, e in enumerate(image_classes) if e == x]
            tp = [[x,y] for x in indices_sketch for y in indices_image]
            image_sketch_hash = image_sketch_hash + tp

        refined_image_sketch_hash = []
        for pair in image_sketch_hash:
            refined_image_sketch_hash.append([sketch_arr[pair[0]],image_arr[pair[1]]])

        self.X_train = refined_image_sketch_hash
        labels = [x[0].split('/')[0] for x in refined_image_sketch_hash]

        values = array(labels)

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        self.y_train = integer_encoded

    def __getitem__(self, index):
        sketch = Image.open(self.sketch_path + self.X_train[index][0])
        image = Image.open(self.image_path + self.X_train[index][1])
        #img.show()
        sketch = sketch.convert('RGB')
        image = image.convert('RGB')
        if self.transform is not None:
            sketch = self.transform(sketch)
            image = self.transform(image)

        label = torch.from_numpy(self.y_train[index])
        return sketch, image, label

    def __len__(self):
        return len(self.X_train)
