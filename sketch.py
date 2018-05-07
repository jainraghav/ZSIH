
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
from Datasets.load_dataset import SketchDataset


IMG_PATH = 'TU-Berlin-sketch/'
IMG_EXT = '.png'
TRAIN_DATA = 'TU-Berlin-sketch/filelist.txt'

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

def test(test_loader):
    model.eval()
    for data in test_loader:
        images, labels = data
        output,attn = model(Variable(images).cuda())
        tflog(attn,images)
        _,predicted = torch.max(output,1)
        break

def tflog(attn,images):
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

def main():
    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_train = SketchDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)

    train_loader = DataLoader(dset_train,batch_size=256,shuffle=True,num_workers=2,pin_memory=True)
    test_loader = DataLoader(dset_train,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)

    num_epochs = 30
    log_dir = 'Log/'
    logger = LogMetric.Logger(log_dir, force=True)
    for epoch in range(1, num_epochs):
        train(epoch,train_loader,logger)
        test(test_loader)
    print("End of training !")

if __name__ == '__main__':
    main()
