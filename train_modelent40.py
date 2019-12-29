from __future__ import print_function
from datasets import PartDataset, ModelNetDataset
from pointnet import PointNetAuto
from tqdm import tqdm
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

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='auto',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--jitter', type=bool, default=False)
parser.add_argument('--sample', type=bool, default=False)


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ModelNetDataset('./data', split='train', jitter=opt.jitter, sample=opt.sample)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = ModelNetDataset('./data', split='test', jitter=opt.jitter, sample=opt.sample)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x:'\033[94m' + x + '\033[0m'

num_points = dataset[0].shape[0]
classifier = PointNetAuto(num_points=num_points, k=3)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
if torch.cuda.is_available():
    classifier.cuda()

num_batch = len(dataset)/opt.batchSize

loss_fn = torch.nn.MSELoss()
classifier = classifier.train()

for epoch in range(opt.nepoch):
    pbar = tqdm(total=num_batch)
    total_loss = 0
    
    for i, data in enumerate(dataloader, 0):
        points = data
        points = Variable(points)
        points = points.transpose(2,1)
        
        if torch.cuda.is_available():
            points = points.cuda()
            
        optimizer.zero_grad()
        
        pred, _ = classifier(points)
        loss = loss_fn(pred, points)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.update(1)

    pbar.close()
    
    j, data = next(enumerate(testdataloader, 0))
    points = data
    points = Variable(points)
    points = points.transpose(2,1)
    if torch.cuda.is_available():
        points = points.cuda()
        
    classifier = classifier.eval()
    pred, _ = classifier(points)
    
    loss = loss_fn(pred, points)    
    print('[%d] %s loss: %f train loss: %f \n' 
          % (epoch, blue('test'), loss.item(), total_loss / int(num_batch)))

    torch.save(classifier.state_dict(), '%s/model_%d.pth' % (opt.outf, epoch))