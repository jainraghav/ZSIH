"""
Network models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd.variable import Variable

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class SparseMM(torch.autograd.Function):

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.sigmoid(x)

class Net(nn.Module):
    def __init__(self,M):
        super(Net, self).__init__()
        # Layers to follow here
        self.attention_hl = 380
        self.encoder_hl = 512
        #AlexNet
        # original_model = models.vgg16(pretrained=True)
        original_model = models.alexnet(pretrained=True)
        new_features = nn.Sequential(*list(original_model.features.children())[:-2])

        original_model.features = new_features
        new_model = original_model.features

        #Attention model
        attn_model = nn.Sequential(nn.Conv2d(256,self.attention_hl, kernel_size=1),
                                   nn.Conv2d(self.attention_hl,1, kernel_size=1)
                                   )
        #Binary encoder
        H = self.encoder_hl
        binary_encoder = nn.Sequential(
            nn.Linear(256, H),
            nn.Linear(H, M)
        )
        self.alex = new_model
        self.attn = attn_model
        self.encoder = binary_encoder

    def forward(self, x):
        # Functions on layers to follow here
        #print(x.size()) #Size here is [256,3,224,224]
        #import pdb; pdb.set_trace()
        x = self.alex(x)
        #print(x.size()) #Size here is [256,256,13,13]

        '''Attention Model after here'''
        #pdb.set_trace()
        attn_mask = self.attn(x)
        attn_mask = attn_mask.view(attn_mask.size(0),-1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0),1,x.size(2),x.size(3))

        #print(x.size(),attn_mask.size()) #Size here is [256,256,13,13]
        #print(attn_mask)
        # x = x*attn_mask
        x = x*attn_mask
        # print(x.size())

        x = x.sum(2).sum(2)
        # print(x.size()) #Size here is [256,256]
        multimodal_input = x

        x = self.encoder(x)
        # print(x.size()) #Size here is [256,250]

        return x,multimodal_input,attn_mask
