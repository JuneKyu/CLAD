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

        self.train, self.test = mnist_dataset(root_dir)

        self.train_x = None
        #  self.train_y = None
        self.test_in_x = None
        #  self.test_in_y = None
        self.test_out_x = None
        #  self.test_out_y = None

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


def mnist_dataset(directory='../data'):

    # pre-calculated mean and std for each classes
    mean_std = [([0.17339932511920858], [0.3477179330970789]),
                ([0.07599864255996079], [0.2442815198263185]),
                ([0.14897512882292896], [0.32592346621877916]),
                ([0.14153014329202565], [0.3179185701004565]),
                ([0.1213655909128458], [0.2974842743686564]),
                ([0.12874939405756766], [0.3035884950833772]),
                ([0.13730177522174805], [0.314897464665824]),
                ([0.11452769775108766], [0.2916958093452544]),
                ([0.1501559818936975], [0.3252599552279906]),
                ([0.12258994285224596], [0.29863753746886956])]

    normal_mean = 0
    normal_std = 0
    for i in config.normal_clas_index_list:
        normal_mean += mean_std[i][0]
        normal_mean += mean_std[i][1]
    normal_mean = normal_mean / len(config.normal_class_index_list)
    normal_std = normal_std / len(config.normal_class_index_list)

    mnist_data_path = directory
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
        #  transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    train = MNIST(mnist_data_path,
                  download=True,
                  train=True,
                  transform=mnist_transform)
    test = MNIST(mnist_data_path,
                 download=True,
                 train=False,
                 transform=mnist_transform)

    return train, test
