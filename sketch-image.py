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
import math
import re
import pdb
import numpy as np
from PIL import Image
import csv
import timeit
import pandas as pd
from numpy import array
import scipy.sparse as sp

import gensim.downloader as api

#Own modules
from Logger import LogMetric
from options import Options
from Models import net
from Datasets.load_SketchImagepairs import SketchImageDataset
from preprocessing import divide_into_sets_disjointclasses
from preprocessing import preprocess_folder

class kproduct(nn.Module):
    def __init__(self):
        super(kproduct, self).__init__()
        self.Wsk = nn.Linear(256,256)
        self.Wim = nn.Linear(256,256)

    def kprod(self,A,B):
        prod = []
        assert(len(A)==len(B))
        A1 = A.view(-1,1).repeat(1,256).view(250,-1)
        B1 = B.repeat(1,256).view(250,-1)
        prod = A1*B1
        # print(prod.size())
        return prod

    def forward(self, x, y):
        x = self.Wsk(x)
        y = self.Wim(y)
        xy = self.kprod(x,y)
        return F.relu(xy)

class Word2Vec():
    def __init__(self):
        print("Loading Word2Vec Model")
        info = api.info()
        self.model = api.load("word2vec-google-news-300")
        print("Model loaded !")

    def vec(self,word):
        return self.model.word_vec(word)

    def most_similar_vec(self,vec):
        return self.model.most_similar(vec)

    def find_closest(self,word):
        return self.model.similar_by_word(word)

    def distance(self,a,b):
        return math.exp(-1*self.model.distance(a,b)/0.1)

class SemanticDecoder(nn.Module):
    def __init__(self,hashlen):
        super(SemanticDecoder, self).__init__()
        self.mu = nn.Linear(hashlen,300)
        self.sigma = nn.Linear(hashlen,300)
        # self.std = torch.from_numpy(np.random.normal(0,1),size=300).cuda().double()
    def forward(self,b,labels):
        #import pdb; pdb.set_trace()
        mean = self.mu(b)
        var = self.sigma(b)*self.sigma(b)
        diag_id = torch.eye(var.size(1)).cuda().double()
        batch_var = diag_id*var.unsqueeze(2).expand(*var.size(),var.size(1))

        diff = labels - mean
        inv = [t.diag().reciprocal().diag() for t in torch.functional.unbind(batch_var)]
        final = torch.stack(inv)

        

        psb = distribution_p1
        return psb

def lossfn(psb,gy,fx,hash,bin_hash,hashlen):
    import pdb; pdb.set_trace()
    h1 = bin_hash.detach()
    mse = nn.MSELoss()
    imgloss = mse(fx,h1)
    sketchloss = mse(gy,h1)
    bce = nn.BCELoss()
    qloss = bce(hash,bin_hash)
    res = (qloss+torch.sum(psb)) + 1/hashlen*(imgloss+sketchloss)
    return res

def gnn(word2vec,batch_labels,input,hashlen):
    batch_size = len(batch_labels)
    adj = np.zeros((batch_size,batch_size))

    for i in range(batch_size):
        for j in range(batch_size):
            adj[i][j] = word2vec.distance(batch_labels[i],batch_labels[j])

    adj_d = torch.from_numpy(adj).cuda()
    features = input.type(torch.cuda.DoubleTensor)

    return adj_d,features

def stochastic_neurons(hash):
    random = torch.rand(hash.shape[1]).double().cuda()
    random_distribution = random.repeat(hash.shape[0],1)

    on = torch.ones(hash.shape[0],hash.shape[1]).double().cuda()
    zr = torch.zeros(hash.shape[0],hash.shape[1]).double().cuda()
    binarized_hash = torch.where(hash>=random_distribution,on,zr)

    return binarized_hash,random

def save_checkpoint(state, checkpoint):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)

def train(hashlen,decoder,graph_model,word2vec_model,model_s,model_i,concat,optimizer,epoch,train_loader,logger,sketchdir,imgdir):
    model_s.train()
    model_i.train()
    concat.train()
    graph_model.train()
    decoder.train()

    for batch_idx, (data_s, data_i, target) in enumerate(train_loader):
        data_s, data_i, target = Variable(data_s).cuda(), Variable(data_i).cuda(), target

        optimizer.zero_grad()

        gy, multimodal_s, attn_s = model_s(data_s)
        fx, multimodal_i, attn_i = model_i(data_i)

        save_checkpoint({'epoch': epoch + 1,'state_dict': model_s.state_dict(),'optim_dict' : optimizer.state_dict()}, sketchdir)
        save_checkpoint({'epoch': epoch + 1,'state_dict': model_i.state_dict(),'optim_dict' : optimizer.state_dict()}, imgdir)

        combined = concat(multimodal_s,multimodal_i)
        # here combined is of size([250,65536])

        adj,features = gnn(word2vec_model,target,combined,hashlen)
        output_hash = graph_model(features,adj)
        binarized_hash,random_distribution = stochastic_neurons(output_hash)

        # Create w2v from target
        semantics = []
        for x in target:
            semantics.append(word2vec_model.vec(x))
        semantics = torch.from_numpy(np.asarray(semantics)).double().cuda()
        # -------------------------------------------------------------------

        psb = decoder(output_hash,semantics)
        loss = lossfn(psb,gy.double(),fx.double(),output_hash,binarized_hash,hashlen)

        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_s), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        # logger.add_scalar('loss_train', float(loss.item()))
        # logger.step()

def test(test_loader,sketchdir,imgdir):
    model_s = torch.load(sketchdir)
    model_i = torch.load(imgdir)

    for batch_idx, (data_s, data_i, target) in enumerate(test_loader):
        gy,_,blah1 = model_s(data_s-0.5)
        fx,_,blah2 = model_s(data_i-0.5)
        sketchb_hash = (np.sign(gy)+1)/2
        imageb_hash = (np.sign(gy)+1)/2


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

def sketch_image_encoder(epochs,logdir,sketch_path,image_path,sketch_data,image_data,hashlen):

    SKETCH_TRAIN_DATA, SKETCH_TEST_DATA, SKETCH_VALID_DATA = divide_into_sets_disjointclasses(sketch_data,sketch_path)
    IMG_TRAIN_DATA, IMG_TEST_DATA, IMG_VALID_DATA = divide_into_sets_disjointclasses(image_data,image_path)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_train = SketchImageDataset(SKETCH_TRAIN_DATA, IMG_TRAIN_DATA, sketch_path, image_path, transformations)
    dset_test = SketchImageDataset(SKETCH_TEST_DATA, IMG_TEST_DATA, sketch_path, image_path, transformations)
    dset_valid = SketchImageDataset(SKETCH_VALID_DATA, IMG_VALID_DATA, sketch_path, image_path, transformations)

    train_loader = DataLoader(dset_train,batch_size=250,shuffle=True,num_workers=2,pin_memory=True)
    test_loader = DataLoader(dset_test,batch_size=250,shuffle=True,num_workers=2,pin_memory=True)
    valid_loader = DataLoader(dset_valid,batch_size=250,shuffle=True,num_workers=2,pin_memory=True)
    sketch_model = net.Net(hashlen).cuda()
    image_model = net.Net(hashlen).cuda()
    concat = kproduct().cuda()
    word2vec_model = Word2Vec()
    graph_model = net.GCN(nfeat=65536, nhid=1024, nclass=hashlen, dropout=0.5).cuda().double()
    decoder = SemanticDecoder(hashlen).cuda().double()

    optimizer = optim.Adam(list(sketch_model.parameters()) + list(image_model.parameters()) + list(concat.parameters()) + list(graph_model.parameters()) + list(decoder.parameters()), lr=0.01)
    # optimizer = optim.SGD(list(sketch_model.parameters()) + list(image_model.parameters()) + list(concat.parameters()) + list(graph_model.parameters()) + list(decoder.parameters()), lr=0.01, momentum=0.5)

    sketchdir = "saved_models/sketch/"
    imgdir = "saved_models/image/"

    num_epochs = epochs
    log_dir = logdir
    logger = LogMetric.Logger(log_dir, force=True)

    for epoch in range(1, num_epochs):
        train(hashlen,decoder,graph_model,word2vec_model,sketch_model,image_model,concat,optimizer,epoch,train_loader,logger,sketchdir,imgdir)
        #validate(valid_loader)
        #log(logging_loader,logger)
    test(test_loader,sketchdir,imgdir)
    print("End of training !")

def main():
    args = Options().parse()
    classes = preprocess_folder(args.img_path,args.img_all_data)
    classes =  preprocess_folder(args.sketch_path,args.sketch_all_data)

    sketch_image_encoder(args.epochs,args.logdir,args.sketch_path,args.img_path,args.sketch_all_data,args.img_all_data,args.hashcode_length)

if __name__ == '__main__':
    main()