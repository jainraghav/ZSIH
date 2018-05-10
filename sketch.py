#Python modules
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torchvision import models

import os
import pdb
from PIL import Image
import pandas as pd
from numpy import array

#Own modules
from Logger import LogMetric
from Models import net
from options import Options
from Datasets.load_sketchdataset import SketchDataset

from preprocessing import divide_into_sets_allclasses
from preprocessing import preprocess_img_folder
from train_test import train,validate

def log(test_loader,logger):
    model.eval()
    for data in test_loader:
        images, labels = data
        output,multimodal_input,attn = model(Variable(images).cuda())
        tflog(attn,images,logger)
        _,predicted = torch.max(output,1)
        break

def tflog(attn,images,logger):
    b = vutils.make_grid(attn[0], normalize=True, scale_each=True)
    bn = (b-b.min())
    bn = bn/bn.max()
    c = vutils.make_grid(images[0], normalize=True, scale_each=True)
    d = vutils.make_grid(attn[81], normalize=True, scale_each=True)
    dn = (d-d.min())
    dn = dn/dn.max()
    e = vutils.make_grid(images[81], normalize=True, scale_each=True)
    logger.add_image('Attention_mask1', bn)
    logger.add_image('Image1', c)
    logger.add_image('Attention_mask2', dn)
    logger.add_image('Image2', e)

def train_encoder_network_as_classifier(ALL_DATA,IMG_PATH):

    TRAIN_DATA,TEST_DATA,VALID_DATA = divide_into_sets_allclasses(ALL_DATA,IMG_PATH)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_train = SketchDataset(TRAIN_DATA,IMG_PATH,transformations)
    dset_test = SketchDataset(TEST_DATA,IMG_PATH,transformations)
    dset_valid = SketchDataset(VALID_DATA,IMG_PATH,transformations)

    train_loader = DataLoader(dset_train,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)
    test_loader = DataLoader(dset_test,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)
    valid_loader = DataLoader(dset_valid,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)

    logging_loader = DataLoader(dset_train,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)

    model = net.Net().cuda()
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_epochs = 10
    log_dir = 'Log/'
    logger = LogMetric.Logger(log_dir, force=True)
    for epoch in range(1, num_epochs):
        train(model,optimizer,epoch,train_loader,logger)
        train_acc = validate(model,train_loader,logger)
        print("Training Accuracy=",train_acc,"%")
        valid_acc = validate(model,valid_loader,logger)
        print("Validation Accuracy=",valid_acc,"%")
        test_acc = validate(model,test_loader,logger)
        print("Test Accuracy=",test_acc,"%")

        logger.add_scalar('Train_Accuracy', train_acc)
        logger.add_scalar('Test_Accuracy', test_acc)
        logger.add_scalar('Valid_Accuracy', valid_acc)
        #validate(valid_loader)
        #log(logging_loader,logger)
    print("End of training !")

def main_classifier():
    #Zero-Shot Learning divide
    #divide_into_sets_disjointclasses(ALL_DATA)

    # Parse options
    args = Options().parse()
    #Train Image Model
    preprocess_img_folder(args.img_path,args.img_all_data)
    train_encoder_network_as_classifier(args.img_all_data,args.img_path)

    #Train Sketch Model
    # train_encoder_network_as_classifier(args.sketch_all_data,args.sketch_path)

if __name__ == '__main__':
    main_classifier()
