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
from torch.distributions.multivariate_normal import MultivariateNormal

import os
import time
import math
import re
import numpy as np
from PIL import Image
import csv
import timeit
import pandas as pd
from numpy import array
import scipy.sparse as sp
import multiprocessing
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
import gensim.downloader as api

#Own modules
from Logger import LogMetric
from options import Options
from Models import net
from Datasets.load_SketchImagepairs import SketchImageDataset,Datapoints
from preprocessing import divide_into_sets_disjointclasses
from preprocessing import preprocess_folder
from test import testmap

class kproduct(nn.Module):
    def __init__(self):
        super(kproduct, self).__init__()
        self.Wsk = nn.Linear(256,256)
        self.Wim = nn.Linear(256,256)

    def kprod(self,A,B):
        prod = []
        assert(len(A)==len(B))
        A1 = A.view(-1,1).repeat(1,256).view(len(A),-1)
        B1 = B.repeat(1,256).view(len(B),-1)
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
    def forward(self,b,labels):
        mean = self.mu(b)
        var = self.sigma(b)*self.sigma(b)
        diag_id = torch.eye(var.size(1)).cuda().double()
        batch_var = diag_id*var.unsqueeze(2).expand(*var.size(),var.size(1))
        m = MultivariateNormal(mean, batch_var)

        # should encorporate labels here somehow

        # rep = labels.mul(var).add_(mean)
        return m.rsample(),mean,var
        # return rep

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def lossfn(labels,psb,mean,var,gy,fx,hash,bin_hash,hashlen):
    import pdb; pdb.set_trace()
    imgloss = F.mse_loss(fx,bin_hash.detach())
    sketchloss = F.mse_loss(gy,bin_hash.detach())

    #qloss and ploss are need to be changed
    qloss = F.binary_cross_entropy_with_logits(hash,bin_hash)

    ploss = F.binary_cross_entropy_with_logits(psb,labels)

    # ploss = F.cross_entropy(psb,hash.detach())
    # ploss = cross_entropy(psb,labels)

    res = (qloss+ploss)+(imgloss+sketchloss)
    return res,imgloss,sketchloss,qloss,ploss

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def qb(binarized_hash,hash):
    result = hash.pow(binarized_hash).mul((1-hash).pow(1-binarized_hash))
    return torch.prod(result,1)

def gnn(word2vec,batch_labels,input,hashlen):
    batch_size = len(batch_labels)
    adj = np.zeros((batch_size,batch_size))
    #t0 = time.time()
    for i in range(batch_size):
        for j in range(batch_size):
            adj[i][j] = word2vec.distance(batch_labels[i],batch_labels[j])
    #print(time.time()-t0)
    adj_d = torch.from_numpy(adj).cuda()
    features = torch.div(input,input.max()).double()
    return adj_d,features

def stochastic_neurons(hash):
    random_distribution = torch.rand(hash.shape[0],hash.shape[1]).double().cuda()
    # random_distribution = random.repeat(hash.shape[0],1)
    on = torch.ones(hash.shape[0],hash.shape[1]).double().cuda()
    zr = torch.zeros(hash.shape[0],hash.shape[1]).double().cuda()
    binarized_hash = torch.where(hash>=random_distribution,on,zr)

    return binarized_hash,random_distribution

def save_checkpoint(state, checkpoint, epoch):
    filepath = os.path.join(checkpoint, str(epoch)+'epoch.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)

def decay_lr(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train(hashlen,decoder,graph_model,word2vec_model,model_s,model_i,enc_i,enc_s,concat,optimizer,epoch,train_loader,logger):
    model_s.train()
    model_i.train()
    concat.train()
    graph_model.train()
    decoder.train()
    enc_i.train()
    enc_s.train()

    for batch_idx, (data_s, data_i, target) in enumerate(train_loader):
        data_s, data_i, target = Variable(data_s).cuda(), Variable(data_i).cuda(), target

        optimizer.zero_grad()
        multimodal_s, attn_s = model_s(data_s)
        multimodal_i, attn_i = model_i(data_i)

        import pdb; pdb.set_trace()

        fx = enc_i(multimodal_i)
        gy = enc_s(multimodal_s)
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

        psb,mean,var = decoder(output_hash,semantics)
        loss,a,b,c,d = lossfn(semantics,psb,mean,var,gy.double(),fx.double(),output_hash,binarized_hash,hashlen)

        loss.backward()
        optimizer.step()

        if batch_idx%4==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_s), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        logger.add_scalar('loss_train', float(loss.item()))
        logger.step()
    print(a.item(),b.item(),c.item(),d.item())

def sketch_image_encoder(epochs,logdir,sketch_path,image_path,sketch_data,image_data,hashlen):

    SKETCH_TRAIN_DATA, SKETCH_TEST_DATA, SKETCH_VALID_DATA = divide_into_sets_disjointclasses(sketch_data,sketch_path)
    IMG_TRAIN_DATA, IMG_TEST_DATA, IMG_VALID_DATA = divide_into_sets_disjointclasses(image_data,image_path)

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_train = SketchImageDataset(SKETCH_TRAIN_DATA, IMG_TRAIN_DATA, sketch_path, image_path, transformations)
    dset_test_s = Datapoints(SKETCH_TEST_DATA, sketch_path, transformations)
    dset_test_i = Datapoints(IMG_TEST_DATA, image_path, transformations)
    dset_valid_s = Datapoints(SKETCH_VALID_DATA, sketch_path, transformations)
    dset_valid_i = Datapoints(IMG_VALID_DATA, image_path, transformations)

    word2vec_model = Word2Vec()

    train_loader = DataLoader(dset_train,batch_size=250,shuffle=True,num_workers=4,pin_memory=True)
    test_sketch_loader = DataLoader(dset_test_s,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    test_image_loader = DataLoader(dset_test_i,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    # test_loader = DataLoader(dset_test,batch_size=250,shuffle=True,num_workers=4,pin_memory=True)
    # valid_loader = DataLoader(dset_valid,batch_size=250,shuffle=True,num_workers=4,pin_memory=True)
    sketch_model = net.Net().cuda()
    image_model = net.Net().cuda()

    enc_i = net.Encoder(hashlen).cuda()
    enc_s = net.Encoder(hashlen).cuda()

    concat = kproduct().cuda()
    graph_model = net.GCN(nfeat=65536, nhid=1024, nclass=hashlen, dropout=0.4).cuda().double()
    decoder = SemanticDecoder(hashlen).cuda().double()

    optimizer = optim.Adam(list(sketch_model.parameters()) + list(image_model.parameters()) + list(enc_i.parameters()) + list(enc_s.parameters()) + list(concat.parameters()) + list(graph_model.parameters()) + list(decoder.parameters()), lr=0.01)
    # optimizer = optim.SGD(list(sketch_model.parameters()) + list(image_model.parameters()) + list(concat.parameters()) + list(graph_model.parameters()) + list(decoder.parameters()), lr=0.01, momentum=0.5)

    sketchdir = "saved_models/sketch/"
    imgdir = "saved_models/image/"

    num_epochs = epochs
    log_dir = logdir
    logger = LogMetric.Logger(log_dir, force=True)

    for epoch in range(1, num_epochs):
        train(hashlen,decoder,graph_model,word2vec_model,sketch_model,image_model,enc_i,enc_s,concat,optimizer,epoch,train_loader,logger)
        #optimizer = decay_lr(optimizer,epoch)
        save_checkpoint({'epoch': epoch,'state_dict_1': sketch_model.state_dict(),'state_dict_2': enc_s.state_dict(),'optim_dict' : optimizer.state_dict()}, sketchdir, epoch)
        save_checkpoint({'epoch': epoch,'state_dict_1': image_model.state_dict(), 'state_dict_2': enc_i.state_dict(),'optim_dict' : optimizer.state_dict()}, imgdir, epoch)
        test_map = testmap(test_image_loader,test_sketch_loader,sketch_model,image_model,enc_i,enc_s)
        logger.add_scalar('mAP', test_map)
    print("End of training !")

def main():
    args = Options().parse()
    classes = preprocess_folder(args.img_path,args.img_all_data)
    classes =  preprocess_folder(args.sketch_path,args.sketch_all_data)
    sketch_image_encoder(args.epochs,args.logdir,args.sketch_path,args.img_path,args.sketch_all_data,args.img_all_data,args.hashcode_length)

if __name__ == '__main__':
    main()
