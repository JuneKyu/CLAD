#!/usr/bin/env python3
# -*- codeing: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
import time
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import config

import pdb

# -------------------------
# Naive ML-based classifier
# -------------------------


# KNN
def KNN_classifier(n_neighbors, train_data, train_label):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(train_data, train_label)
    return model


# svm
def SVM_classifier(gamma, C, train_data, train_label):
    model = svm.SVC(gamma=gamma, C=C)
    model.fit(train_data, train_label)
    return model


# -------------------------
# Neural-Network-based classifier
# -------------------------

#  def FC3_classifier():

#  def CNN_classifier():
#  super(CNN_classifier, self).__init__()


# for linear classifier
class BinaryClassification(nn.Module):
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
        predicted = []
        with torch.no_grad():
            out_features = self.linear(input_dim)
            predict_sm = F.softmax(out_features)
            predict_sm = predict_sm.detach().cpu().numpy()
            for i in range(len(predict_sm)):
                predicted.append(
                    np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted


# for text based data
# TODO: add validataion set
def Linear_classifier(train_data, train_cluster, n_epochs, lr):

    _, input_size = train_data.shape

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    #  test_data = scaler.fit_transform(test_data)

    train_data = torch.from_numpy(train_data.astype(np.float32)).cuda(
        config.device)
    #  test_data = torch.from_numpy(test_data.astype(np.float32))
    #  pdb.set_trace()
    train_cluster = torch.from_numpy(train_cluster).cuda(config.device)
    #  test_cluster = torch.from_numpy(test_cluster)

    model = BinaryClassification(input_dim=input_size).cuda(config.device)

    criterion = nn.CrossEntropyLoss()  # Log Softmax + ClassNLL Loss
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = np.zeros(n_epochs)
    #  test_losses = np.zeros(n_epochs)

    for iter_ in range(n_epochs):
        outputs = model(train_data)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, train_cluster)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: loss can be ploted
        #  train_losses[iter_] = loss.item()

        if (iter_ + 1) % 10 == 0:
            print("In this epoch {}/{}, Training loss: {}".format((iter_ + 1),
                                                                  n_epochs,
                                                                  loss.item()))

    return model


# for text classifier
class Text_GRUNet(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layers=2,
                 drop_prob=0.2):
        super(Text_GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          n_layers,
                          batch_first=True,
                          dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size,
                            self.hidden_dim).zero_().to(config.device)
        return hidden


def GRU_text_classifier(train_data, train_label, test_data, test_label,
                        n_epochs, lr):

    input_size = len(train_data[0])
    train_label = torch.tensor(train_label)
    train_loader = DataLoader(TensorDataset(train_data, train_label),
                              shuffle=True,
                              batch_size=config.text_classifier_batch_size,
                              drop_last=True)

    model = Text_GRUNet(input_dim=input_size,
                        hidden_dim=config.text_classifier_hidden_size,
                        output_dim=config.text_classifier_output_size)
    model.to(config.device)
    #  criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    # gradient clipping if needed
    model.train()
    print("training text classifier...")

    num_epoch = config.text_classifier_epoch
    batch_size = config.text_classifier_batch_size
    epoch_times = []
    pdb.set_trace()
    for epoch in range(1, num_epoch + 1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            h = h.data
            model.zero_grad()

            out, h = model(x.to(config.device).float(), h)
            loss = criterion(out, label.to(config.device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print(
                    "Epoch {}.....Step: {}/{}..... Average Loss for Epoch: {}".
                    format(epoch, counter, len(train_loader),
                           avg_loss / counter))
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(
            epoch, num_epoch, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(
            str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    pdb.set_trace()

    model.eval()
    print("evaluating text classifier...")
    outputs = []
    targets = []
    start_time = time.clock()
    #  test_loader = DataLoader(TensorDataset(test_data, test_label))
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        lab = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(config.device).float(), h)
        sm = nn.Softmax(out.cpu().detach.numpy())
        pdb.set_trace()
        outputs.append()

        target.append(test_label[i].cpu().detach.numpy())
    print("Evaluation Time: {}".format(str(time.clock() - start_time)))
    #  for i in range(len(outputs)):

    return model
    #  model = Text_GRUNet(input_dim=input_siz)


# for WideResNet block unit
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes,
                               out_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(self.equalInOut and out or x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.add((not self.equalInOut) and self.convShortcut(x) or x,
                         out)


# for WideResNet block unit
class NetworkBlock(nn.Module):
    def __init__(self,
                 nb_layers,
                 in_planes,
                 out_planes,
                 block,
                 stride,
                 dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes,
                      i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], nChannels[1], block, 1,
                               dropRate)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
                                   dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                   dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 3,
                                   dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def WideResNet_classifier():
    print("")
