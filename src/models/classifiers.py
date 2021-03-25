#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet, BasicBlock

import torch.nn.functional as F


import numpy as np

import config

import pdb

# -------------------------
# Neural-Network-based classifier
# -------------------------


# for linear classifier
class Linear_Model(nn.Module):
    def __init__(self, input_dim):
        super(Linear_Model).__init__()
        self.out_features_dim = config.cluster_num
        self.linear = nn.Linear(input_dim, self.out_features_dim)

    def forward(self, x):
        if (len(x.shape) >= 4):
            x = torch.reshape(x, (len(x), -1))
        elif (len(x.shape) >= 3):
            x = torch.reshape(x, (1, -1))
        return self.linear(x)

    def predict(self, x):
        if (len(x.shape) >= 4):
            x = torch.reshape(x, (len(x), -1))
        elif (len(x.shape) >= 3):
            x = torch.reshape(x, (1, -1))
        dataloader = DataLoader(x, batch_size=128)
        predicted = []
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                out_features = self.linear(data)
                predict_sm = F.softmax(out_features)
                predict_sm = predict_sm.detach().cpu().numpy()
                for i in range(len(predict_sm)):
                    predicted.append(
                        np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted


class FC3_Model(nn.Module):
    def __init__(self, input_dim):
        super(FC3_Model).__init__()
        self.out_features_dim = config.cluster_num
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.linear3 = nn.Linear(input_dim, self.out_features_dim)

    def forward(self, x):
        if (len(x.shape) >= 4):
            x = torch.reshape(x, (len(x), -1))
        elif (len(x.shape) >= 3):
            x = torch.reshape(x, (1, -1))
        output1 = self.linear1(x)
        output2 = self.linear2(output1)
        output3 = self.linear3(output2)
        return output3

    def predict(self, x):
        if (len(x.shape) >= 4):
            x = torch.reshape(x, (len(x), -1))
        elif (len(x.shape) >= 3):
            x = torch.reshape(x, (1, -1))
        dataloader = DataLoader(x, batch_size=128)
        predicted = []
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                out_features1 = self.linear1(data)
                out_features2 = self.linear2(out_features1)
                out_features3 = self.linear3(out_features2)
                predict_sm = F.softmax(out_features3)
                predict_sm = predict_sm.detach().cpu().numpy()
                for i in range(len(predict_sm)):
                    predicted.append(
                        np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted


# for image data
class CNN_Model(nn.Module):
    def __init__(self, batch_size, channels, height, width, is_rgb=True):
        super(CNN_Model, self).__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.out_features_dim = config.cluster_num

        self.layer = nn.Sequential(nn.Conv2d(self.channels, 16, 3, padding=1),
                                   nn.BatchNorm2d(16), nn.ReLU(),
                                   nn.Conv2d(16, 32, 3, padding=1),
                                   nn.BatchNorm2d(32), nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(32, 64, 3, padding=1),
                                   nn.BatchNorm2d(64), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * (self.height // 4) * (self.width // 4), 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64,
                                                     self.out_features_dim))

    def forward(self, x):
        if (len(x.shape) < 4):
            x = x.reshape(1, self.channels, self.height, self.width)
        out = self.layer(x)
        if (x.shape[0] == 1):
            out = out.view(1, -1)
        else:
            out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out

    def predict(self, x):
        dataloader = DataLoader(x, batch_size=128)
        predicted = []
        with torch.no_grad():
            # data size adjustment
            for _, image in enumerate(dataloader):
                out = self.layer(image)
                out = out.view(image.shape[0], -1)
                out = self.fc_layer(out)
                predict_sm = F.softmax(out)
                predict_sm = predict_sm.detach().cpu().numpy()
                for i in range(len(predict_sm)):
                    predicted.append(
                        np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted


# for image data
class CNNLarge_Model(nn.Module):
    def __init__(self, batch_size, channels, height, width, is_rgb=True):
        super(CNNLarge_Model, self).__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.out_features_dim = config.cluster_num
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=self.channels,
                      out_channels=self.channels * 8,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(self.channels * 8), nn.ELU(),
            nn.Conv2d(in_channels=self.channels * 8,
                      out_channels=self.channels * 16,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(self.channels * 16), nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=self.channels * 16,
                      out_channels=self.channels * 32,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(self.channels * 32),
            nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_layer = nn.Sequential(
            nn.Linear(
                self.channels * 32 * (self.height // 4) * (self.width // 4),
                128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64,
                                                     self.out_features_dim))

    def forward(self, x):
        if (len(x.shape) < 4):
            x = x.reshape(1, self.channels, self.height, self.width)
        out = self.layer(x)
        if (x.shape[0] == 1):
            out = out.view(1, -1)
        else:
            out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out

    def predict(self, x):
        dataloader = DataLoader(x, batch_size=128)

        predicted = []
        with torch.no_grad():
            for _, image in enumerate(dataloader):
                out = self.layer(image)
                out = out.view(image.shape[0], -1)
                out = self.fc_layer(out)
                predict_sm = F.softmax(out)
                predict_sm = predict_sm.detach().cpu().numpy()
                #  for i in range(len(predict_sm)):
                predicted.append(np.where(predict_sm == max(predict_sm))[0][0])
        return predicted
    

class ResNet_Model(ResNet):
    def __init__(self, batch_size, channels, height, width, is_rgb=True):
        # [2, 2, 2, 2] => num of each layers
        super(ResNet_Model, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=config.cluster_num)
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        #  self.is_rgb = is_rgb
        self.out_features_dim = config.cluster_num
        if not is_rgb:
            self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
 
    def forward(self, x):
        if (len(x.shape) < 4):
            x = x.reshape(1, self.channels, self.height, self.width)
        out = super(ResNet_Model, self).forward(x)
        return out

    def predict(self, x):
        dataloader = DataLoader(x, batch_size=128)
        predicted = []
        with torch.no_grad():
            for _, image in enumerate(dataloader):
                out = super(ResNet_Model, self).forward(image)
                predict_sm = F.softmax(out)
                predict_sm = predict_sm.detach().cpu().numpy()
                for i in range(len(predict_sm)):
                    predicted.append(
                        np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
                #  predicted.append(np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted
