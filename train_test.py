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

#Own modules
from Logger import LogMetric


def train(model,optimizer,epoch,train_loader,logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        target = target.type('torch.cuda.LongTensor')
        # print(data.size(),target.size())
        optimizer.zero_grad()
        #import pdb; pdb.set_trace()
        output,multimodal_input,attn = model(data)
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

def validate(model,valid_loader,logger):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            labels = labels.cuda()
            output,multimodal_input,attn = model(Variable(images).cuda())
            _,predicted = torch.max(output,1)
            for i in range(len(images)):
                a,b = predicted[i].item(),labels[i].item()
                total+=1
                if a==b:
                    correct+=1
    acc = (correct/total)*100
    return acc
