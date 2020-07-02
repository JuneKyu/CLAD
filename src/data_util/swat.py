#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from scipy import signal
import sklearn.preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset
import torch
import config

import pdb


class SWaT_Dataset(object):
    """Docstring for SWaT_Dataset. """
    def __init__(self, root_dir: str):

        self._root_dir = root_dir
        self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = swat_dataset(
            root_dir)

    def get_dataset(self):

        self.train_x = torch.tensor(self.train_x).float()
        self.train_y = torch.tensor(self.train_y).float()
        self.val_x = torch.tensor(self.val_x).float()
        self.val_y = torch.tensor(self.val_y).float()
        self.test_x = torch.tensor(self.test_x).float()
        self.test_y = torch.tensor(self.test_y).float()

        train = TensorDataset(self.train_x, self.train_y)
        val = TensorDataset(self.val_x, self.val_y)
        test = TensorDataset(self.test_x, self.test_y)

        dataset = {"train": train, "val": val, "test": test}

        return dataset


def swat_dataset(directory='../data'):

    np.random.seed(777)

    data_path = os.path.join(directory, 'swat_data')

    train_path = os.path.join(data_path, 'SWaT_Dataset_Normal_v0.csv')
    test_path = os.path.join(data_path, 'SWaT_Dataset_Attack_v0.csv')

    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)

    # frequency information to concatenate
    freq_train_x = get_freq_data(data=train,
                                 read_size=config.read_size,
                                 window_size=config.window_size,
                                 freq_select_list=config.swat_selected_list)
    freq_test_x = get_freq_data(data=test,
                                read_size=config.read_size,
                                window_size=config.window_size,
                                freq_select_list=config.swat_selected_list)

    x_col = list(train.columns)[:-1]
    y_col = list(train.columns)[-1]
    train_x = train[x_col]
    train_y = train[y_col]
    test_x = test[x_col]
    test_y = test[y_col]
    train_y[train_y == 'Normal'] = 0
    train_y[train_y == 'Attack'] = 1
    test_y[test_y == 'Normal'] = 0
    test_y[test_y == 'Attack'] = 1

    train_x_modify_swat, test_x_modify_swat = PCA_preprocessing_modify(
        scaler='standard',
        train_x=train_x,
        test_x=test_x,
        selected_dim=config.swat_raw_selected_dim)
    train_x_modify_swat_freq, test_x_modify_swat_freq = PCA_preprocessing_modify(
        scaler='standard',
        train_x=freq_train_x,
        test_x=freq_test_x,
        selected_dim=config.swat_freq_selected_dim)

    train_concat_x = np.concatenate(
        (train_x_modify_swat, train_x_modify_swat_freq), 1)
    test_concat_x = np.concatenate(
        (test_x_modify_swat, test_x_modify_swat_freq), 1)

    val_x = test_concat_x.copy()
    val_y = test_y.copy()

    return train_concat_x, train_y, val_x, val_y, test_concat_x, test_y


def get_freq_data(data, read_size, window_size, freq_select_list):

    data = data[freq_select_list]
    f, t, Sxx = signal.spectrogram(data.values[:, 0], 1)

    tp = np.zeros((data.shape[0], read_size), dtype=float)

    tp[0:int(t[0]), :] = Sxx[0:read_size, 0]

    for i in range(0, Sxx.shape[1] - 1):
        tp[int(t[i]):int(t[i + 1])] = np.sum(Sxx[0:read_size, i])

    freq_data = np.copy(tp)
    for feature in range(1, data.shape[1]):
        f, t, Sxx = signal.spectrogram(data.values[:, feature],
                                       1,
                                       window=signal.tukey(window_size))
        tp = np.zeros((data.shape[0], read_size), dtype=float)
        tp[0:int(t[0]), :] = Sxx[0:read_size, 0]

        for i in range(0, Sxx.shape[1] - 1):
            tp[int(t[i]):int(t[i + 1]), :] = Sxx[0:read_size, i]
        freq_data = np.concatenate(
            (freq_data.reshape(data.shape[0], -1), tp.reshape(
                data.shape[0], -1)),
            axis=1)

    return freq_data


def PCA_preprocessing_modify(scaler, train_x, test_x, selected_dim):

    if scaler == 'standard':
        transformer = sklearn.preprocessing.StandardScaler()
        transformer.fit(train_x)
    elif scaler == 'normalizer':
        transformer = sklearn.preprocessing.Normalizer()
        transformer.fit(train_x)

    train_x = transformer.transform(train_x)
    test_x = transformer.transform(test_x)

    pca = PCA(n_components=np.shape(train_x)[1])
    pca.fit(train_x)

    train_x = pca.transform(train_x)[:, selected_dim]
    test_x = pca.transform(test_x)[:, selected_dim]

    return train_x, test_x
