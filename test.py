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
np.seterr(divide='ignore', invalid='ignore')
from numpy import array

from Models import net
from options import Options
from Datasets.load_SketchImagepairs import SketchImageDataset,Datapoints

def testmap(test_img_loader,test_sketch_loader,model_s,model_i,f,g):
    all_image_hashes = []
    img_labels=[]
    with torch.no_grad():
        for batch_idx,(data_i,label,_) in enumerate(test_img_loader):
            data_i = data_i.cuda()
            rel_i,mask_i = model_i(data_i)
            # m = nn.BatchNorm1d(256, affine=False).cuda()
            # zz = m(rel_i)
            fx = f(rel_i-0.5)
            fxd = fx.cpu().detach().numpy()
            imageb_hash = (np.sign(fxd)+1)/2

            all_image_hashes.extend(imageb_hash)
            img_labels.extend(label)

        all_sketch_hashes = []
        sketch_labels=[]
        for batch_idx, (data_s, target,_) in enumerate(test_sketch_loader):
            data_s = data_s.cuda()
            rel_s,mask_s = model_s(data_s)
            # m = nn.BatchNorm1d(256, affine=False).cuda()
            # yy = m(rel_s)
            gy = g(rel_s-0.5)
            gyd = gy.cpu().detach().numpy()
            sketchb_hash = (np.sign(gyd)+1)/2

            all_sketch_hashes.extend(sketchb_hash)
            sketch_labels.extend(target)

    # hamm_d = cdist(all_sketch_hashes, all_image_hashes, 'euclidean')
    hamm_d = cdist(all_sketch_hashes, all_image_hashes, 'hamming')
    #Sort-Slice-mAP@100
    hamm_d.sort(axis=1)
    # import pdb; pdb.set_trace()
    str_sim = (np.expand_dims(sketch_labels, axis=1) == np.expand_dims(img_labels, axis=0)) * 1
    nq = str_sim.shape[0]
    num_cores = min(multiprocessing.cpu_count(), 32)

    hamm_d = hamm_d[0:nq,0:100] #to calc mAP@100
    str_sim = str_sim[0:nq,0:100]
    hamm_d = 1 - hamm_d
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], hamm_d[iq]) for iq in range(100))
    map_ = np.mean(aps)
    print('Mean Average Precision {map:.4f}'.format(map=map_))
    return map_

if __name__ == '__main__':
    args = Options().parse()

    transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    dset_test_s = Datapoints(args.sketch_path + "filelist-test.txt", args.sketch_path, transformations)
    dset_test_i = Datapoints(args.img_path + "filelist-test.txt", args.img_path, transformations)

    test_sketch_loader = DataLoader(dset_test_s,batch_size=64,shuffle=True,num_workers=4,pin_memory=True)
    test_image_loader = DataLoader(dset_test_i,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)

    sketch_model = net.Net().cuda()
    image_model = net.Net().cuda()
    f = net.Encoder(args.hashcode_length).cuda()
    g = net.Encoder(args.hashcode_length).cuda()

    #change the checkpoint you want to load here...
    checkpoint_s = torch.load(args.sketch_model+"1epoch.pth.tar")
    checkpoint_i = torch.load(args.image_model+"1epoch.pth.tar")

    sketch_model.load_state_dict(checkpoint_s['state_dict_1'])
    image_model.load_state_dict(checkpoint_i['state_dict_1'])

    f.load_state_dict(checkpoint_i['state_dict_2'])
    g.load_state_dict(checkpoint_s['state_dict_2'])

    map_test = testmap(test_image_loader,test_sketch_loader,sketch_model,image_model,f,g)
