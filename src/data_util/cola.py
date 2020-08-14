#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wget
import os
import zipfile

import pandas as pd
from torch.utils.data import TensorDataset

import config
from .embeddings import preprocess_with_avg_bert
from .embeddings import preprocess_with_s_bert

import pdb


class CoLA_Dataset(object):
    """Docstring for CoLA_Dataset. """
    def __init__(self, root_dir: str):

        self._root_dir = root_dir
        self.train, self.val, self.test = cola_dataset(root_dir)
        #  self.input_ids = []

        self.train_x = None
        self.train_y = None

        self.val_x = None
        self.val_y = None

        self.test_x = None
        self.test_y = None

    def preprocess(self):

        which_embedding = config.embedding
        assert which_embedding in config.implemented_nlp_embeddings

        print("embedding with {} embedding".format(which_embedding))

        if which_embedding == 'avg_glove':
            print("not implemented yet")
        elif which_embedding == 'avg_bert':
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                                    preprocess_with_avg_bert(self.train, self.val, self.test)
        elif which_embedding == 's_bert':
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                                    preprocess_with_s_bert(self.train, self.val, self.test)

    def get_dataset(self):

        train = TensorDataset(self.train_x, self.train_y)
        val = TensorDataset(self.val_x, self.val_y)
        test = TensorDataset(self.test_x, self.test_y)

        dataset = {"train": train, "val": val, "test": test}

        return dataset


def cola_dataset(directory='../data'):

    cola_data_path = os.path.join(directory, 'cola_public')

    if not os.path.exists(cola_data_path):
        print("Downloading CoLA dataset...")
        os.mkdir(cola_data_path)
        url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
        cola_zip = zipfile.ZipFile(
            os.path.join(cola_data_path, 'cola_public_1.1.zip'))
        cola_zip.extractall(cola_data_path)
        os.remove(os.path.join(cola_data_path, 'cola_public_1.1.zip'))

    cola_train_data_path = os.path.join(cola_data_path,
                                        "raw/in_domain_train.tsv")
    train = pd.read_csv(
        cola_train_data_path,
        delimiter='\t',
        header=None,
        names=['sentence_source', 'label', 'label_notes', 'sentence'])

    # extract only normal cases for the cola sentences
    train = train[train.label != 0]

    cola_val_data_path = os.path.join(cola_data_path, "raw/in_domain_dev.tsv")
    val = pd.read_csv(
        cola_val_data_path,
        delimiter='\t',
        header=None,
        names=['sentence_source', 'label', 'label_notes', 'sentence'])

    cola_test_data_path = os.path.join(cola_data_path,
                                       "raw/out_of_domain_dev.tsv")
    test = pd.read_csv(
        cola_test_data_path,
        delimiter='\t',
        header=None,
        names=['sentence_source', 'label', 'label_notes', 'sentence'])

    return train, val, test
