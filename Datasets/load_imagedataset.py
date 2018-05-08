"""
Image Data Loader
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
