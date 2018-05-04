
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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Own modules
from Logger import LogMetric


IMG_PATH = 'TU-Berlin-sketch/'
IMG_EXT = '.png'
TRAIN_DATA = 'TU-Berlin-sketch/filelist.txt'

# Testing code is commented out
class SketchDataset(Dataset):
    """Arguments:
        CSV filepaths
        Path to img folder
        Extension of images
        PIL Transforms
    """
    def __init__(self, csv_path, img_path, img_ext, transform=None):

        tmp_df = pd.read_csv(csv_path)
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['ImagePath']

        arr = tmp_df['ImagePath'].str.partition('/')[0].values.tolist()
        values = array(arr)

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        # binary encode

        #print(integer_encoded)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded).astype(float)
        #print(onehot_encoded)
        self.y_train = integer_encoded
        #self.y_train = onehot_encoded
        #print(type(self.y_train))
        #print(self.y_train[0])

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index])
        #img.show()
        #x = plt.imread(self.img_path + self.X_train[index])
        #plt.imshow(x)
        #plt.show()
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)



transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])

dset_train = SketchDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)

train_loader = DataLoader(dset_train,
                          batch_size=256,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True # CUDA only
                         )
test_loader = DataLoader(dset_train,
                          batch_size=256,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=True # CUDA only
                         )

#AlexNet
original_model = models.alexnet(pretrained=True)
#print(original_model)
new_features = nn.Sequential(*list(original_model.features.children())[:-2])
#RELU is the last layer which is debatable to keep or not

original_model.features = new_features
new_model = original_model.features
#print(new_model)


#Attention model
attn_hidden_layer=380
attn_model = nn.Sequential(nn.Conv2d(256,attn_hidden_layer, kernel_size=1),
                          nn.Conv2d(attn_hidden_layer,1, kernel_size=1)
                          )
#print(attn_model)

#Binary encoder
H=256
binary_encoder = nn.Sequential(
    nn.Linear(256, H),
    nn.Linear(H, 250)
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Layers to follow here
        self.alex = new_model
        self.attn = attn_model

        self.encoder = binary_encoder

    def forward(self, x):
        # Functions on layers to follow here
        # print(x.size()) #Size here is [256,3,224,224]

        x = self.alex(x)
        # print(x.size()) #Size here is [256,256,13,13]

        '''Attention Model after here'''
        #pdb.set_trace()
        attn_mask = self.attn(x)
        attn_mask = attn_mask.view(attn_mask.size(0),-1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0),1,x.size(2),x.size(3))
 
        #print(x.size(),attn_mask.size()) #Size here is [256,256,13,13]
        #print(attn_mask)

        x = x+x*attn_mask
        # print(x.size())

        x = x.sum(2).sum(2)
        # print(x.size()) #Size here is [256,256]

        x = self.encoder(x)
        # print(x.size()) #Size here is [256,250]

        return x,attn_mask

model = Net().cuda()
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.5)

def train(epoch):
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

def test():
	model.eval()
	for data in test_loader:
		images, labels = data
		output,attn = model(Variable(images).cuda())
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
		
		_,predicted = torch.max(output,1)
		break

		
num_epochs = 10
log_dir = 'Log/'
logger = LogMetric.Logger(log_dir, force=True) 
for epoch in range(1, num_epochs):    
    train(epoch)
    test()
#test()

print("End of training !")
