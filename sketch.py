
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

def divide_into_sets(all_classes,trainp,validp,testp):
    all = list(all_classes)
    train_set = all[:int(trainp*len(all))]
    valid_set = all[int(trainp*len(all))+1:int(trainp*len(all))+1+int(validp*len(all))]
    test_set = all[int(trainp*len(all))+1+int(validp*len(all)):]
    return train_set,valid_set,test_set

mod_alex,attn_model,binary_encoder = preprocess_networklayers(attention_hl=380,encoder_hl=512)
model = net.Net(mod_alex,attn_model,binary_encoder).cuda()
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.5)

def train(epoch,train_loader,logger):
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
        logger.step()

def validate(valid_loader):
    model.eval()
    correct=0
    total=0
    for data in valid_loader:
        images, labels = data
        labels = labels.cuda()
        output,attn = model(Variable(images).cuda())
        _,predicted = torch.max(output,1)
        c = (predicted == labels).squeeze()
        #print(c,c.size())
        correct+=(c.sum(0).sum(0)).item()

    print("Accuracy=",correct/12000)

def test(test_loader):
    model.eval()
    for data in test_loader:
        images, labels = data
        output,attn = model(Variable(images).cuda())
        _,predicted = torch.max(output,1)

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

def create_seperate_files(train_set,valid_set,test_set):

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

def main():
    line_prepender(ALL_DATA,"ImagePath")
    tmp_df = pd.read_csv(ALL_DATA)
    arr = tmp_df['ImagePath'].str.partition('/')[0].values.tolist()
    all_classes = set(arr)
    train_set,valid_set,test_set = divide_into_sets(all_classes,0.6,0.2,0.2)
    create_seperate_files(train_set,valid_set,test_set)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_train = SketchDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)
    dset_test = SketchDataset(TEST_DATA,IMG_PATH,IMG_EXT,transformations)
    dset_valid = SketchDataset(VALID_DATA,IMG_PATH,IMG_EXT,transformations)

    train_loader = DataLoader(dset_train,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)
    test_loader = DataLoader(dset_test,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)
    valid_loader = DataLoader(dset_valid,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)


    logging_loader = DataLoader(dset_train,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)

    num_epochs = 30
    log_dir = 'Log/'
    logger = LogMetric.Logger(log_dir, force=True)
    for epoch in range(1, num_epochs):
        train(epoch,train_loader,logger)
        validate(train_loader)
        #validate(valid_loader)
        #test(test_loader)
        log(logging_loader,logger)
    print("End of training !")

if __name__ == '__main__':
    main()
