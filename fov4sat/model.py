"""This module defines FoveaNet4Sat as a neural network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cconf


class Foveanet(nn.Module):
    def __init__(self, channels):
        super(Foveanet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, cconf.cfg.kernels[0],
                               padding=cconf.cfg.paddings[0])
        self.conv2 = nn.Conv2d(32, 32, cconf.cfg.kernels[1],
                               padding=cconf.cfg.paddings[1])
        self.conv3 = nn.Conv2d(32, 32, cconf.cfg.kernels[2],
                               padding=cconf.cfg.paddings[2])
        self.conv4 = nn.Conv2d(32, 256, cconf.cfg.kernels[3],
                               padding=cconf.cfg.paddings[3])
        self.conv5 = nn.Conv2d(256, 512, cconf.cfg.kernels[4],
                               padding=cconf.cfg.paddings[4])
        self.conv6 = nn.Conv2d(512, 256, cconf.cfg.kernels[5],
                               padding=cconf.cfg.paddings[5])
        self.conv7 = nn.Conv2d(256, 256, cconf.cfg.kernels[6],
                               padding=cconf.cfg.paddings[6])
        self.conv8 = nn.Conv2d(256, 1, cconf.cfg.kernels[7],
                               padding=cconf.cfg.paddings[7])
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    ef forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        if cconf.cfg.poolingMax is True:
            x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = self.dropout(x)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv8(x))
        return x
