#!/usr/bin/env python3
# -*- codeing: utf-8 -*-

import os

from .swat import SWaT_Dataset
from .cola import CoLA_Dataset
from .reuters import Reuters_Dataset
from config import implemented_datasets


def load_dataset(dataset_name, data_path, normal_class_index):

    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'swat':
        print("loading swat dataset...")
        swat_dataset = SWaT_Dataset(root_dir = data_path)
        #  print("preprocessing...")
        #  swat_dataset.preprocess()
        dataset = swat_dataset.get_dataset()

    elif dataset_name == 'wadi':
        print("loading wadi dataset is not implemented yet")

    elif dataset_name == 'cola':
        print("loading cola dataset...")
        cola_dataset = CoLA_Dataset(root_dir = data_path)
        print("preprocessing...")
        cola_dataset.preprocess()
        dataset = cola_dataset.get_dataset()

    elif dataset_name == 'reuters':
        print("loading reuters dataset...")
        reuters_dataset = Reuters_Dataset(root_dir = data_path)
        print("preprocessing...")
        reuters_dataset.preprocess()
        dataset = reuters_dataset.get_binary_labeled_data(normal_class_index = normal_class_index)

    return dataset

