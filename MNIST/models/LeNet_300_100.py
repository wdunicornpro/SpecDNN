from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
from newLayers import *
from prune import mask

class LeNet_300_100(nn.Module):
    def __init__(self, prune='',mask=[]):
        super(LeNet_300_100, self).__init__()
        self.prune = prune
        self.mask = mask
        self.ip1 = nn.Linear(28*28, 300)
        self.relu_ip1 = nn.ReLU(inplace=True)
        if self.prune == 'node':
            self.prune_ip1 = MaskLayer(300)
        self.ip2 = nn.Linear(300, 100)
        self.relu_ip2 = nn.ReLU(inplace=True)
        if self.prune == 'node':
            self.prune_ip2 = MaskLayer(100)
        self.ip3 = nn.Linear(100, 10)
        return

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        if self.mask:
            self.ip1.weight = mask(self.mask[0],self.ip1.weight)
        # print(self.ip1.weight[:,self.mask[0]].mean())
        x = self.ip1(x)
        x = self.relu_ip1(x)
        if self.prune == 'node':
            x = self.prune_ip1(x)
        if self.mask:
            self.ip2.weight = mask(self.mask[1],self.ip2.weight)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        if self.prune == 'node':
            x = self.prune_ip2(x)
        if self.mask:
            self.ip3.weight = mask(self.mask[2],self.ip3.weight)
        x = self.ip3(x)
        x = torch.sigmoid(x)
        return x
