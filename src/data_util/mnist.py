#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import TensorDataset

from data_util.utils import divide_data_label

import config
import pdb


class MNIST_Dataset(object):
    """Docstring for MNIST_Dataset. """
    def __init__(self, root_dir: str):

        self.dec_train, self.dec_test, self.train, self.test = mnist_dataset(
            root_dir)

        self.train_x = None
        #  self.train_y = None
        self.test_in_x = None
        #  self.test_in_y = None
        self.test_out_x = None
        #  self.test_out_y = None

    def get_dataset(self):

        self.dec_train_x, self.dec_train_y, _, _ = divide_data_label(
            self.dec_train, train=True)
        self.train_x, _, _, _ = divide_data_label(self.train, train=True)
        self.test_in_x, _, self.test_out_x, _ = divide_data_label(
            self.dec_test, train=False)

        self.dec_train_x = torch.tensor(self.dec_train_x)
        self.dec_train_y = torch.tensor(self.dec_train_y)
        self.train_x = torch.tensor(self.train_x)
        self.test_in_x = torch.tensor(self.test_in_x)
        self.test_out_x = torch.tensor(self.test_out_x)
        dataset = {
            "dec_train": self.dec_train_x,
            "dec_train_y": self.dec_train_y,
            "train": self.train_x,
            "test_in": self.test_in_x,
            "test_out": self.test_out_x
        }

        return dataset


def _transformation(img):
    return torch.ByteTensor(torch.ByteStorage.from_buffer(
        img.tobytes())).float() * 0.02


def mnist_dataset(directory='../data'):

    mnist_data_path = directory
    dec_img_transform = transforms.Compose(
        [transforms.Lambda(_transformation)])
    dec_train = MNIST(mnist_data_path,
                      download=True,
                      train=True,
                      transform=dec_img_transform)
    dec_test = MNIST(mnist_data_path,
                     download=True,
                     train=False,
                     transform=dec_img_transform)

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    train = MNIST(mnist_data_path,
                  download=True,
                  train=True,
                  transform=mnist_transform)
    test = MNIST(mnist_data_path,
                 download=True,
                 train=False,
                 transform=mnist_transform)

    return dec_train, dec_test, train, test
