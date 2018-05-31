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

import multiprocessing
from joblib import Parallel, delayed
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

import pdb
import numpy as np
from numpy import array

from Models import net
from options import Options
from Datasets.load_SketchImagepairs import SketchImageDataset,Datapoints

def testmap(test_img_loader,test_sketch_loader,model_s,model_i):
    all_image_hashes = []
    img_labels=[]
    for batch_idx,(data_i,label) in enumerate(test_img_loader):
        data_i = data_i.cuda()
        fx,_,blah = model_i(data_i-0.5)
        fxd = fx.cpu().detach().numpy()
        imageb_hash = (np.sign(fxd)+1)/2
        # import pdb; pdb.set_trace()
        all_image_hashes.extend(imageb_hash)
        img_labels.extend(label)

    all_sketch_hashes = []
    sketch_labels=[]
    for batch_idx, (data_s, target) in enumerate(test_sketch_loader):
        data_s = data_s.cuda()
        gy,_,blah1 = model_s(data_s-0.5)
        gyd = gy.cpu().detach().numpy()
        sketchb_hash = (np.sign(gyd)+1)/2
        all_sketch_hashes.extend(sketchb_hash)
        sketch_labels.extend(target)

    hamm_d = cdist(all_sketch_hashes, all_image_hashes, 'hamming')

    str_sim = (np.expand_dims(sketch_labels, axis=1) == np.expand_dims(img_labels, axis=0)) * 1
    nq = str_sim.shape[0]
    num_cores = min(multiprocessing.cpu_count(), 32)
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], hamm_d[iq]) for iq in range(nq))
    map_ = np.mean(aps)
    print('Mean Average Precision {map:.4f}'.format(map=map_))

if __name__ == '__main__':
    args = Options().parse()

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_test_s = Datapoints(args.sketch_path + "filelist-test.txt", args.sketch_path, transformations)
    dset_test_i = Datapoints(args.img_path + "filelist-test.txt", args.img_path, transformations)

    test_sketch_loader = DataLoader(dset_test_s,batch_size=64,shuffle=True,num_workers=4,pin_memory=True)
    test_image_loader = DataLoader(dset_test_i,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)

    sketch_model = net.Net(args.hashcode_length).cuda()
    image_model = net.Net(args.hashcode_length).cuda()

    checkpoint_s = torch.load("saved_models/sketch/1epoch.pth.tar")
    checkpoint_i = torch.load("saved_models/image/1epoch.pth.tar")

    sketch_model.load_state_dict(checkpoint_s['state_dict'])
    image_model.load_state_dict(checkpoint_i['state_dict'])

    test(test_image_loader,test_sketch_loader,sketch_model,image_model)
