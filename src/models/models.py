#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import config

# -------------------------
# Neural-Network-based classifier
# -------------------------


# for linear classifier
class LinearClassification(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.out_features_dim = config.cluster_num
        self.linear = nn.Linear(input_dim, self.out_features_dim)
        #  self.softmax = nn.Softmax(self.out_features_dim)

    def forward(self, input_dim):
        if (len(input_dim.shape) >= 2):
            input_dim = torch.reshape(input_dim, (len(input_dim), -1))
        #  return torch.unsqueeze(self.linear(input_dim), 0)
        return self.linear(input_dim)

    def predict(self, input_dim):
        if (len(input_dim.shape) >= 2):
            input_dim = torch.reshape(input_dim, (len(input_dim), -1))

        predicted = []
        with torch.no_grad():
            out_features = self.linear(input_dim)
            predict_sm = F.softmax(out_features)
            predict_sm = predict_sm.detach().cpu().numpy()
            for i in range(len(predict_sm)):
                predicted.append(
                    np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted


class FC3Classification(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.out_features_dim = config.cluster_num
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.linear3 = nn.Linear(input_dim, self.out_features_dim)

    def forward(self, input_dim):
        if (len(input_dim.shape) >= 2):
            input_dim = torch.reshape(input_dim, (len(input_dim), -1))
        output1 = self.linear1(input_dim)
        output2 = self.linear2(output1)
        output3 = self.linear3(output2)
        return output3
        #  return torch.unsqueeze(output3, 0)

    def predict(self, input_dim):
        if (len(input_dim.shape) >= 2):
            input_dim = torch.reshape(input_dim, (len(input_dim), -1))
        predicted = []
        with torch.no_grad():
            out_features1 = self.linear1(input_dim)
            out_features2 = self.linear2(out_features1)
            out_features3 = self.linear3(out_features2)
            predict_sm = F.softmax(out_features3)
            predict_sm = predict_sm.detach().cpu().numpy()
            for i in range(len(predict_sm)):
                predicted.append(
                    np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted


# for image data
class CNNClassification(nn.Module):
    def __init__(self, batch_size, is_rgb=True):
        super(CNNClassification, self).__init__()
        if (is_rgb):
            self.color_factor = 3
        else:
            self.color_factor = 1
        self.out_features_dim = config.cluster_num
        self.batch_size = batch_size
        self.layer = nn.Sequential(
            nn.Conv2d(self.color_factor, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_layer = nn.Sequential(nn.Linear(64 * 7 * 7, 128),
                                      nn.BatchNorm1d(128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.BatchNorm1d(64),
                                      nn.ReLU(),
                                      nn.Linear(64, self.out_features_dim))

    def forward(self, x):
        if (len(x.shape) < 4): x = x.reshape(1, 1, 28, 28)
        out = self.layer(x)
        if (x.shape[0] == 1):
            out = out.view(1, -1)
        else:
            out = out.view(self.batch_size, -1)
        out = self.fc_layer(out)
        return out

    def predict(self, input_dim):
        predicted = []
        with torch.no_grad():
            # data size adjustment
            out = self.layer(input_dim)
            out = out.view(input_dim.shape[0], -1)
            out = self.fc_layer(out)
            predict_sm = F.softmax(out)
            predict_sm = predict_sm.detach().cpu().numpy()
            for i in range(len(predict_sm)):
                predicted.append(
                    np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted
