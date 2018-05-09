"""
Network models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

class Net(nn.Module):
    def __init__(self,new_alexnet,attention_model,encoder):
        super(Net, self).__init__()
        # Layers to follow here
        self.alex = new_alexnet
        self.attn = attention_model
        self.encoder = encoder

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
        x = x*attn_mask
        # print(x.size())

        x = x.sum(2).sum(2)
        # print(x.size()) #Size here is [256,256]
        multimodal_input = x

        x = self.encoder(x)
        # print(x.size()) #Size here is [256,250]

        return x,multimodal_input,attn_mask
