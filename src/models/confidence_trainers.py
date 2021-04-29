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
from models.classifiers import Linear_Model, FC3_Model, CNN_Model, CNNLarge_Model, ResNet_Model
from tqdm import tqdm

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
    train_cluster = torch.from_numpy(train_cluster).cuda(config.device)

    model = Linear_Model(input_dim=input_size)
    model = nn.DataParallel(model).cuda(config.device)
    criterion = nn.CrossEntropyLoss()  # Log Softmax + ClassNLL Loss
    optimizer = Adam(model.parameters(), lr=lr)

    for iter_ in range(n_epochs):
        outputs = model(train_data)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, train_cluster)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (iter_ + 1) % 10 == 0:
            print("Epoch {}/{}, Training loss: {}".format(
                (iter_ + 1), n_epochs, loss.item()))
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
    model = FC3_Model(input_dim=input_size)
    model = nn.DataParallel(model).cuda(config.device)
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
            print("Epoch {}/{}, Training loss: {}".format(
                (iter_ + 1), n_epochs, loss.item()))
    return model


# batch_size, isRGB needs to be specified
def CNN_classifier(train_data,
                   train_cluster,
                   n_epochs,
                   lr,
                   batch_size=100,
                   is_rgb=False):

    input_size = train_data.shape[0]
    channels = train_data.shape[1]
    height = train_data.shape[2]
    width = train_data.shape[3]
    train_cluster = torch.from_numpy(train_cluster).cuda(config.device)
    train = TensorDataset(train_data, train_cluster)
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    model = CNN_Model(batch_size, channels, height, width, is_rgb=is_rgb)
    model = model.cuda(config.device)
    model = nn.DataParallel(model)
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

        print("Epoch {}/{}, Training loss: {}".format((iter_ + 1), n_epochs,
                                                      loss.item()))
    return model


def CNN_large_classifier(train_data,
                         train_cluster,
                         n_epochs,
                         lr,
                         batch_size=100,
                         is_rgb=False):

    input_size = train_data.shape[0]
    channels = train_data.shape[1]
    height = train_data.shape[2]
    width = train_data.shape[3]
    train_cluster = torch.from_numpy(train_cluster).cuda(config.device)
    train = TensorDataset(train_data, train_cluster)
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    model = CNN_Model(batch_size, channels, height, width, is_rgb=is_rgb)
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
        print("Epoch {}/{}, Training loss: {}".format((iter_ + 1), n_epochs,
                                                      loss.item()))
    return model


def ResNet_classifier(train_data,
                      train_cluster,
                      n_epochs,
                      lr,
                      batch_size=100,
                      is_rgb=False):
    input_size = train_data.shape[0]
    channels = train_data.shape[1]
    height = train_data.shape[2]
    width = train_data.shape[3]
    train_cluster = torch.from_numpy(train_cluster).cuda(config.device)
    train = TensorDataset(train_data, train_cluster)
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    model = ResNet_Model(batch_size, channels, height, width, is_rgb=is_rgb)
    model = nn.DataParallel(model).cuda(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        data_iterator = tqdm(train_loader,
                             leave='True',
                             unit='batch',
                             postfix={
                                'epoch': epoch,
                                'loss': '%.6f' % 0.0,
                             })
        for _, [image, label] in enumerate(data_iterator):
            image = image.to(config.device)
            label = label.to(config.device)
            outputs = model(image)
            loss = criterion(outputs, label)
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data_iterator.set_postfix(
                epoch=epoch,
                loss='%.6f' % loss_value,
            )
    return model
