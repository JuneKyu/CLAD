#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import wget
import os
import zipfile

import pandas as pd

import config
from data_util.BERT_embedding import preprocess_with_bert


class CoLA_Dataset(object):

    """Docstring for CoLA_Dataset. """

    def __init__(self, root_dir:str):

        self._root_dir = root_dir
        self.train, self.val, self.test = cola_dataset(root_dir)
        self.input_ids = []

        self.train_x = None
        self.train_y = None
                
        self.val_x = None
        self.val_y = None

        self.test_x = None
        self.test_y = None

    def preprocess(self):

        self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                                preprocess_with_bert(
                                        dataset_name = 'cola',
                                        train = self.train,
                                        val = self.val,
                                        test = self.test,
                                        num_labels = config.CoLA_num_of_label,
                                        MAX_LEN = config.MAX_LEN,
                                        num_epochs = config.CoLA_fine_tune_epochs,
                                        batch_size = config.CoLA_fine_tune_batch_size,
                                        )
 
       
    def get_dataset(self):
        return self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y


def cola_dataset(directory = '../data'):
    
    cola_data_path = os.path.join(directory,'cola_public')

    if not os.path.exists(cola_data_path):
        print("Downloading CoLA dataset...")
        os.mkdir(cola_data_path)
        url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
        cola_zip = zipfile.ZipFile(os.path.join(data_path, 'cola_public_1.1.zip'))
        cola_zip.extractall(data_path)

    cola_train_data_path = os.path.join(cola_data_path, "raw/in_domain_train.tsv")
    train = pd.read_csv(cola_train_data_path, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    cola_val_data_path = os.path.join(cola_data_path, "raw/in_domain_dev.tsv")
    val = pd.read_csv(cola_val_data_path, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

    cola_test_data_path = os.path.join(cola_data_path, "raw/out_of_domain_dev.tsv")
    test = pd.read_csv(cola_test_data_path, delimiter='\t',      header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

    return train, val, test

