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
import re
import pdb
import numpy as np
from PIL import Image
import pandas as pd
from numpy import array
from subprocess import call

#Own modules
from Logger import LogMetric
from options import Options
from Models import net
from Datasets.load_SketchImagepairs import SketchImageDataset

from preprocessing import divide_into_sets_disjointclasses
from preprocessing import preprocess_folder

def train(model_s,model_i,optimizer,epoch,train_loader,logger):
    model_s.train()
    model_i.train()
    for batch_idx, (data_s, data_i, target) in enumerate(train_loader):
        '''
        Here target is a list of label strings to be converted to semantic embeddings
        '''
        data_s, data_i, target = Variable(data_s).cuda(), Variable(data_i).cuda(), target
        print(target)
        # import pdb; pdb.set_trace()
        optimizer.zero_grad()
        hash_s, multimodal_s, attn_s = model_s(data_s)
        hash_i, multimodal_i, attn_i = model_i(data_i)

        # input_s = multimodal_s.cpu().detach().numpy()
        # input_i = multimodal_i.cpu().detach().numpy()
        # combined_representation = np.kron(input_s,input_i)


        print(batch_idx)
        #gnn(multimodal1, multimodal2)

        # print(output)
        # target = target.squeeze_()
        # loss = F.cross_entropy(output, target)
        # loss.backward()
        # optimizer.step()
        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        # logger.add_scalar('loss_train', float(loss.item()))
        # logger.step()

def validate(model_s,model_i,valid_loader,logger):
    model_s.eval()
    model_i.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data in valid_loader:
            sketch, image, label = data
            label = label.cuda()
            hash1 ,multimodal_input1 ,attn = model(Variable(sketch).cuda())
            hash2 ,multimodal_input2 ,attn = model(Variable(image).cuda())
    #         _,predicted = torch.max(output,1)
    #         for i in range(len(images)):
    #             a,b = predicted[i].item(),labels[i].item()
    #             total+=1
    #             if a==b:
    #                 correct+=1
    # acc = (correct/total)*100
    return acc

def sketch_image_encoder(epochs,logdir,sketch_path,image_path,sketch_data,image_data):

    SKETCH_TRAIN_DATA, SKETCH_TEST_DATA, SKETCH_VALID_DATA = divide_into_sets_disjointclasses(sketch_data,sketch_path)
    IMG_TRAIN_DATA, IMG_TEST_DATA, IMG_VALID_DATA = divide_into_sets_disjointclasses(image_data,image_path)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_train = SketchImageDataset(SKETCH_TRAIN_DATA, IMG_TRAIN_DATA, sketch_path, image_path, transformations)
    dset_test = SketchImageDataset(SKETCH_TEST_DATA, IMG_TEST_DATA, sketch_path, image_path, transformations)
    dset_valid = SketchImageDataset(SKETCH_VALID_DATA, IMG_VALID_DATA, sketch_path, image_path, transformations)

    train_loader = DataLoader(dset_train,batch_size=250,shuffle=True,num_workers=2,pin_memory=True)
    test_loader = DataLoader(dset_test,batch_size=250,shuffle=True,num_workers=2,pin_memory=True)
    valid_loader = DataLoader(dset_valid,batch_size=250,shuffle=True,num_workers=2,pin_memory=True)

    sketch_model = net.Net().cuda()
    image_model = net.Net().cuda()

    optimizer = optim.SGD(list(sketch_model.parameters()) + list(image_model.parameters()), lr=0.01, momentum=0.5)

    num_epochs = epochs
    log_dir = logdir
    logger = LogMetric.Logger(log_dir, force=True)

    for epoch in range(1, num_epochs):
        train(sketch_model,image_model,optimizer,epoch,train_loader,logger)

        #validate(valid_loader)
        #log(logging_loader,logger)
    print("End of training !")

def main():
    args = Options().parse()
    classes = preprocess_folder(args.img_path,args.img_all_data)
    #replacing special chars with spaces in labels
    refined_classes = [re.sub(r'\W+', ' ', x) for x in classes]
    #print(refined_classes)



    #return
    sketch_image_encoder(args.epochs,args.logdir,args.sketch_path,args.img_path,args.sketch_all_data,args.img_all_data)
    #sketch_image_encoder(args.sketch_path,args.image_path,args.sketch_all_data,args.image_all_data)

if __name__ == '__main__':
    main()
