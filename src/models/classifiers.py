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
from torch.optim import Adam, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

from models.models import LinearClassification
from models.models import FC3Classification
from models.models import CNNClassification
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


# for basic test
def Linear_classifier(train_data, train_cluster, n_epochs, lr):

    if (len(train_data) > 2):
        train_data = torch.reshape(train_data, (len(train_data), -1))

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

    model = LinearClassification(input_dim=input_size).cuda(config.device)
    criterion = nn.CrossEntropyLoss()  # Log Softmax + ClassNLL Loss
    optimizer = Adam(model.parameters(), lr=lr)

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


# for basic test
def FC3_classifier(train_data, train_cluster, n_epochs, lr):

    if (len(train_data) > 2):
        train_data = torch.reshape(train_data, (len(train_data), -1))

    _, input_size = train_data.shape
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = torch.from_numpy(train_data.astype(np.float32)).cuda(
        config.device)
    train_cluster = torch.from_numpy(train_cluster).cuda(config.device)
    model = FC3Classification(input_dim=input_size).cuda(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = np.zeros(n_epochs)

    for iter_ in range(n_epochs):
        outputs = model(train_data)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, train_cluster)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter_ + 1) % 10 == 0:
            print("In this epoch {}/{}, Training loss: {}".format((iter_ + 1),
                                                                  n_epochs,
                                                                  loss.item()))
    return model


# batch_size, isRGB needs to be specified
def CNN_classifier(train_data,
                   train_cluster,
                   n_epochs,
                   lr,
                   batch_size=100,
                   is_rgb=False):

    input_size = train_data.shape[0]
    train_cluster = torch.from_numpy(train_cluster).cuda(config.device)
    train = TensorDataset(train_data, train_cluster)
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    model = CNNClassification(batch_size, is_rgb=is_rgb)
    model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               threshold=0.1,
                                               patience=1,
                                               mode='min')

    for iter_ in range(n_epochs):
        for _, [image, label] in enumerate(train_loader):
            image = image.to(config.device)
            label = label.to(config.device)
            outputs = model(image)

            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(loss)

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


# for image dataset
#  def CNN_classifier(train_data, train_cluster, n_epochs, lr):

#  check if the data is RGB or gray_scale


def Linear_classifier(train_data, train_cluster, n_epochs, lr):

    if (len(train_data) > 2):
        train_data = torch.reshape(train_data, (len(train_data), -1))

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

    model = LinearClassification(input_dim=input_size).cuda(config.device)
    criterion = nn.CrossEntropyLoss()  # Log Softmax + ClassNLL Loss
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = np.zeros(n_epochs)

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
