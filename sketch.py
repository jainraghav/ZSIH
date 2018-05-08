
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
from Datasets.load_sketchdataset import SketchDataset


IMG_PATH = 'TU-Berlin-sketch/'
IMG_EXT = '.png'
ALL_DATA = 'TU-Berlin-sketch/filelist.txt'
TRAIN_DATA = 'TU-Berlin-sketch/filelist-train.txt'
TEST_DATA = 'TU-Berlin-sketch/filelist-test.txt'
VALID_DATA = 'TU-Berlin-sketch/filelist-valid.txt'

def preprocess_networklayers(attention_hl,encoder_hl):
    #AlexNet
    original_model = models.alexnet(pretrained=True)
    #print(original_model)
    new_features = nn.Sequential(*list(original_model.features.children())[:-2])
    #RELU is the last layer which is debatable to keep or not

    original_model.features = new_features
    new_model = original_model.features
    #print(new_model)

    #Attention model
    attn_hidden_layer = attention_hl
    attn_model = nn.Sequential(nn.Conv2d(256,attn_hidden_layer, kernel_size=1),
                              nn.Conv2d(attn_hidden_layer,1, kernel_size=1)
                              )
    #print(attn_model)
    #Binary encoder
    H = encoder_hl
    binary_encoder = nn.Sequential(
        nn.Linear(256, H),
        nn.Linear(H, 250)
    )
    return new_model,attn_model,binary_encoder

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def train(model,optimizer,epoch,train_loader,logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        target = target.type('torch.cuda.LongTensor')
        # print(data.size(),target.size())
        optimizer.zero_grad()
        output,attn = model(data)
        #logger.add_image('Image', output)
        # print(output)
        target = target.squeeze_()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        logger.add_scalar('loss_train', float(loss.item()))

def validate(model,valid_loader,logger):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            labels = labels.cuda()
            output,attn = model(Variable(images).cuda())
            _,predicted = torch.max(output,1)
            for i in range(len(images)):
                a,b = predicted[i].item(),labels[i].item()
                total+=1
                if a==b:
                    correct+=1
    acc = (correct/total)*100
    return acc

def log(test_loader,logger):
    model.eval()
    for data in test_loader:
        images, labels = data
        output,attn = model(Variable(images).cuda())
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

#for zero-shot learning task
def divide_into_sets_disjointclasses(ALL_DATA,trainp=0.6,validp=0.2,testp=0.2):

    tmp_df = pd.read_csv(ALL_DATA)
    arr = tmp_df['ImagePath'].str.partition('/')[0].values.tolist()
    all_classes = set(arr)
    all = list(all_classes)
    train_set = all[:int(trainp*len(all))]
    valid_set = all[int(trainp*len(all))+1:int(trainp*len(all))+1+int(validp*len(all))]
    test_set = all[int(trainp*len(all))+1+int(validp*len(all)):]

    with open(ALL_DATA) as f:
        with open(TRAIN_DATA, "w") as f1:
            f1.write("ImagePath\n")
            for line in f:
                if line.split('/')[0] in train_set:
                    f1.write(line)

    with open(ALL_DATA) as f:
        with open(TEST_DATA, "w") as f1:
            f1.write("ImagePath\n")
            for line in f:
                if line.split('/')[0] in test_set:
                    f1.write(line)

    with open(ALL_DATA) as f:
        with open(VALID_DATA, "w") as f1:
            f1.write("ImagePath\n")
            for line in f:
                if line.split('/')[0] in valid_set:
                    f1.write(line)

#for normal classification learning task
def divide_into_sets_allclasses(ALL_DATA,trainp=0.6,validp=0.2,testp=0.2):
    train_set=[]
    valid_set=[]
    test_set=[]
    tmp_df = pd.read_csv(ALL_DATA)
    classes = list(set(tmp_df['ImagePath'].str.partition('/')[0].values.tolist()))
    arr = tmp_df['ImagePath'].values.tolist()
    for x in classes:
        temp_one_class=[]
        for im in arr:
            if im.split('/')[0]==x:
                temp_one_class.append(im)
        train_set = train_set + temp_one_class[:int(trainp*len(temp_one_class))]
        valid_set = valid_set + temp_one_class[int(trainp*len(temp_one_class)) : int(trainp*len(temp_one_class)) + int(validp*len(temp_one_class))]
        test_set = test_set + temp_one_class[int(trainp*len(temp_one_class)) + int(validp*len(temp_one_class)) : ]

    #print(classes)
    with open(TRAIN_DATA, "w") as f1:
        f1.write("ImagePath\n")
        for line in train_set:
            f1.write(line+"\n")

    with open(TEST_DATA, "w") as f1:
        f1.write("ImagePath\n")
        for line in test_set:
            f1.write(line+"\n")

    with open(VALID_DATA, "w") as f1:
        f1.write("ImagePath\n")
        for line in valid_set:
            f1.write(line+"\n")

def main():

    #divide_into_sets_disjointclasses(ALL_DATA)
    divide_into_sets_allclasses(ALL_DATA)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_train = SketchDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)
    dset_test = SketchDataset(TEST_DATA,IMG_PATH,IMG_EXT,transformations)
    dset_valid = SketchDataset(VALID_DATA,IMG_PATH,IMG_EXT,transformations)

    train_loader = DataLoader(dset_train,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)
    test_loader = DataLoader(dset_test,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)
    valid_loader = DataLoader(dset_valid,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)


    logging_loader = DataLoader(dset_train,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)

    mod_alex,attn_model,binary_encoder = preprocess_networklayers(attention_hl=380,encoder_hl=512)
    model = net.Net(mod_alex,attn_model,binary_encoder).cuda()
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


    num_epochs = 30
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
        logger.step()
        #validate(valid_loader)
        #log(logging_loader,logger)
    print("End of training !")

if __name__ == '__main__':
    main()
