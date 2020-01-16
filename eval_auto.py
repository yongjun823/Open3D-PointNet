from __future__ import print_function
from datasets import PartDataset, ModelNetDataset
from pointnet import PointNetAuto
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
from chamfer_distance import ChamferDistance


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--jitter', type=bool, default=False)
parser.add_argument('--sample', type=bool, default=False)


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

test_dataset = ModelNetDataset('./data', split='test', jitter=opt.jitter, sample=opt.sample)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

origin_dataset = ModelNetDataset('./data', split='test', jitter=False, sample=False)
origindataloader = torch.utils.data.DataLoader(origin_dataset, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))


num_batch = len(test_dataset) // opt.batchSize

blue = lambda x:'\033[94m' + x + '\033[0m'

num_points = test_dataset[0].shape[0]
classifier = PointNetAuto(num_points=num_points, k=3)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

if torch.cuda.is_available():
    classifier.cuda()

chamfer_dist = ChamferDistance()

loss_arr = []
classifier.eval()

with torch.no_grad():
    for test_data, origin_data in zip(testdataloader, origindataloader):
        points = test_data
        points = points.transpose(2,1)
        origin_points = origin_data.transpose(2,1)
        
        if torch.cuda.is_available():
            points = points.cuda()
            origin_points = origin_points.cuda()
            
        classifier = classifier.eval()
        pred, _ = classifier(points)

        dist1, dist2 = chamfer_dist(origin_points, pred)
        
        loss = (torch.sum(dist1)) + (torch.sum(dist2))
        
        loss_arr.append(loss.item())

print('%s loss: %f' % (blue('test'), np.mean(loss_arr)))
print('%s var : %f' % (blue('test'), np.var(loss_arr)))
