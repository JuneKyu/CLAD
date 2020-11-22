#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import TensorDataset

from data_util.utils import divide_data_label

import config
import pdb


class CIFAR10_Dataset(object):
    """Docstring for CIFAR10_Dataset. """
    def __init__(self, root_dir: str):

        #  self.dec_train, self.dec_test, self.train, self.test = cifar10_dataset(
        #      root_dir)
        self.train, self.test = cifar10_dataset(root_dir)
        self.train_x = None
        self.test_in_x = None
        self.test_out_x = None

    def get_dataset(self):
        self.train_x, self.train_y, _, _ = divide_data_label(self.train,
                                                             train=True)
        self.test_in_x, _, self.test_out_x, _ = divide_data_label(self.test,
                                                                  train=False)

        self.train_x = torch.tensor(self.train_x)
        self.train_y = torch.tensor(self.train_y)
        self.test_in_x = torch.tensor(self.test_in_x)
        self.test_out_x = torch.tensor(self.test_out_x)

        dataset = {
            "train_x": self.train_x,
            "train_y": self.train_y,
            "test_in": self.test_in_x,
            "test_out": self.test_out_x
        }

        return dataset


def cifar10_dataset(directory='../data'):

    cifar10_data_path = directory
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        #  transforms.Normalize(mean=[123.3 / 255, 123.0 / 255, 113.9 / 255],
        #                       std=[63.0 / 255, 62.1 / 255, 66.7 / 255.0])
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.243, 0.261])
    ])

    train = CIFAR10(cifar10_data_path,
                    download=True,
                    train=True,
                    transform=cifar_transform)
    test = CIFAR10(cifar10_data_path,
                   download=True,
                   train=False,
                   transform=cifar_transform)

    return train, test
