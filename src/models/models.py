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
        self.out_features_dim = len(config.normal_class_index_list)
        self.linear = nn.Linear(input_dim, self.out_features_dim)
        #  self.softmax = nn.Softmax(self.out_features_dim)

    def forward(self, input_dim):
        return torch.unsqueeze(self.linear(input_dim), 0)

        #  output = self.linear(input_dim)
        #  output = self.softmax(output)
        #  return output
    def predict(self, input_dim):
        if (len(input_dim) > 2):
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
        self.out_features_dim = len(config.normal_class_index_list)
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.linear3 = nn.Linear(input_dim, self.out_features_dim)

    def forward(self, input_dim):
        output1 = self.linear1(input_dim)
        output2 = self.linear2(output1)
        output3 = self.linear3(output2)
        return torch.unsqueeze(output3, 0)

    def predict(self, input_dim):
        if (len(input_dim) > 2):
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
    def __init__(self, batch_size, isRGB=True):
        super(CNNClassification, self).__init__()
        if (isRGB):
            self.color_factor = 3
        else:
            self.color_factor = 1
        self.batch_size = batch_size
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, self.color_factor, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.Conv2d(16, 32, self.color_factor, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, self.color_factor, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_layer = nn.Sequential(nn.Linear(64 * 7 * 7, 127),
                                      nn.BatchNorm1d(128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.BatchNorm1d(64),
                                      nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        out = self.layer(x)
        out = out.view(self.batch_size, -1)
        out = self.fc_layer(out)
        return out
