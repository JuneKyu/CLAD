#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import TensorDataset

import config
import pdb


class MNIST_Dataset(object):
    """Docstring for MNIST_Dataset. """
    def __init__(self, root_dir: str):

        self._root_dir = root_dir
        self.train, self.test = mnist_dataset(root_dir)

        self.train_x = None
        self.train_y = None
        #  self.val_x = None
        #  self.val_y = None
        self.test_in_x = None
        self.test_in_y = None
        self.test_out_x = None
        self.test_out_y = None

    def get_dataset(self):
        #  self.train_x, self.train_y = divide_data_label(self.train, train=True)
        #  self.val_x, self.val_y = divide_data_label(self.val, train=False)
        #  self.test_x, self.test_y = divide_data_label(self.test, train=False)

        self.train_x, self.train_y, _, _ = divide_data_label(self.train,
                                                             train=True)
        self.test_in_x, self.test_in_y, self.test_out_x, self.test_out_y = divide_data_label(
            self.test, train=False)

        self.train_x = torch.tensor(self.train_x)
        self.train_y = torch.tensor(self.train_y)
        #  self.val_x = torch.tensor(self.val_x)
        #  self.val_y = torch.tensor(self.val_y)
        self.test_in_x = torch.tensor(self.test_in_x)
        self.test_in_y = torch.tensor(self.test_in_y)

        self.test_out_x = torch.tensor(self.test_out_x)
        self.test_out_y = torch.tensor(self.test_out_y)
        train = TensorDataset(self.train_x, self.train_y)
        #  val = TensorDataset(self.val_x, self.val_y)
        test_in = TensorDataset(self.test_in_x, self.test_in_y)
        test_out = TensorDataset(self.test_out_x, self.test_out_y)

        dataset = {"train": train, "test_in": test_in, "test_out": test_out}

        return dataset


def divide_data_label(dataset, train=False):
    in_data = []
    out_data = []
    in_labels = []
    out_labels = []
    for _d in dataset:
        data_x = _d[0].numpy()
        data_y = _d[1]

        #  if (data_y in config.normal_class_index_list):
        #      in_data.append(data_x)
        #      in_labels.append(0)
        #  else:
        #      in_data.append(data_x)
        #      in_labels.append(1)

        if (data_y in config.normal_class_index_list):
            in_data.append(data_x)
            in_labels.append(0)
        else:
            if (train): continue
            else:
                out_data.append(data_x)
                out_labels.append(1)

    return in_data, in_labels, out_data, out_labels


def _transformation(img):
    return torch.ByteTensor(torch.ByteStorage.from_buffer(
        img.tobytes())).float() * 0.02


def mnist_dataset(directory='../data'):

    mnist_data_path = directory
    img_transform = transforms.Compose([transforms.Lambda(_transformation)])
    train = MNIST(mnist_data_path,
                  download=True,
                  train=True,
                  transform=img_transform)
    #  val = MNIST(mnist_data_path,
    #  download=True,
    #  train=False,
    #  transform=img_transform)
    test = MNIST(mnist_data_path,
                 download=True,
                 train=False,
                 transform=img_transform)

    return train, test
