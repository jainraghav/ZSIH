"""
Network models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd.variable import Variable

class Net(nn.Module):
    def __init__(self):
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
            nn.Linear(H, 250)
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
