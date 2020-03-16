#!/usr/bin/env python3
# -*- codeing: utf-8 -*-

import os

from data_util.cola import CoLA_Dataset
from config import implemented_datasets


def load_dataset(dataset_name, data_path):

    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'swat':
        print("loading swat dataset is not implemented yet")
    elif dataset_name == 'wadi':
        print("loading wadi dataset is not implemented yet")
    elif dataset_name == 'cola':
        print("loading cola dataset...")
        cola_dataset = CoLA_Dataset(root_dir = data_path)
        print("preprocess with pretrained bert")
        cola_dataset.preprocess()
        return cola_dataset.get_dataset()


