import os
import wget
import pickle
import zipfile
import csv
from shutil import move

import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import TensorDataset

from data_util.utils import divide_data_label

import config
import pdb


class GTSRB_Dataset(object):
    """Docstring for GTSRB_Dataset. """
    def __init__(self, root_dir: str):

        self.train, self.test = gtsrb_dataset(root_dir)
        self.train_x = None
        self.test_in_x = None
        self.test_out_x = None

    def get_dataset(self):
        self.train_x, self.train_y, _, _, = divide_data_label(self.train, train=True)
        self.test_in_x, _, self.test_out_x, _ = divide_data_label(self.test, train=False)

        self.train_x = clean_Nonetypes(self.train_x)
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


def gtsrb_dataset(directory='../data'):
    
    if (os.path.exists(os.path.join(directory, 'GTSRB')) == False):
        gtsrb_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370'
        gtsrb_train_url = os.path.join(gtsrb_url, 'GTSRB_Final_Training_Images.zip')
        gtsrb_test_url = os.path.join(gtsrb_url, 'GTSRB_Final_Test_Images.zip')
        gtsrb_test_gt_url = os.path.join(gtsrb_url, 'GTSRB_Final_Test_GT.zip')
        print("downloading train dataset...")
        wget.download(gtsrb_train_url, out = directory)
        print()
        print("downloading test dataset...")
        wget.download(gtsrb_test_url, out = directory)
        print()
        wget.download(gtsrb_test_gt_url, out = directory)
        print()
        gtsrb_train_zip = zipfile.ZipFile(os.path.join(directory, 'GTSRB_Final_Training_Images.zip'))
        gtsrb_test_zip = zipfile.ZipFile(os.path.join(directory, 'GTSRB_Final_Test_Images.zip'))
        gtsrb_test_gt_zip = zipfile.ZipFile(os.path.join(directory, 'GTSRB_Final_Test_GT.zip'))
        print("extracting train dataset...")
        gtsrb_train_zip.extractall(directory)
        print("extracting test dataset...")
        gtsrb_test_zip.extractall(directory)
        gtsrb_test_gt_zip.extractall(os.path.join(directory, 'GTSRB'))
        os.remove(os.path.join(directory, 'GTSRB_Final_Training_Images.zip'))
        os.remove(os.path.join(directory, 'GTSRB_Final_Test_Images.zip'))
        os.remove(os.path.join(directory, 'GTSRB_Final_Test_GT.zip'))
    
    mean_std_pickle_path = 'gtsrb_mean_std.pkl'
    with open(os.path.join(directory, '../src/data_util/' + mean_std_pickle_path), 'rb') as f:
        mean_std = pickle.load(f)

    normal_mean = [0, 0, 0]
    normal_std = [0, 0, 0]
    for i in config.normal_class_index_list:
        for j in range(3):
            normal_mean[j] += mean_std[i][0][j]
            normal_std[j] += mean_std[i][1][j]

    for k in range(3):
        normal_mean[k] = normal_mean[k] / len(config.normal_class_index_list)
        normal_std[k] = normal_std[k] / len(config.normal_class_index_list)
    
    gtsrb_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    
    train_path = os.path.join(directory, 'GTSRB/Final_Training/Images')
    test_path = os.path.join(directory, 'GTSRB/Final_Test/Images')

    if (os.path.exists(os.path.join(directory, 'GTSRB/Final_Test/Images/00000')) == False):
        divide_test_path(directory)
    train = datasets.ImageFolder(train_path, transform=gtsrb_transform)
    test = datasets.ImageFolder(test_path, transform=gtsrb_transform)
    
    return train, test


def divide_test_path(directory='../data'):

    test_annotations = get_test_labels(os.path.join(directory, 'GTSRB/GT-final_test.csv'))
    test_path = os.path.join(directory, 'GTSRB/Final_Test/Images')
    # make class dirs
    for i in range(43):
        os.mkdir(os.path.join(directory, 'GTSRB/Final_Test/Images/' + "{:05d}".format(i)))
    for annotation in test_annotations:
        path = os.path.join(test_path, annotation[0])
        dest = os.path.join(test_path, '{:05d}'.format(annotation[1]) + '/' + annotation[0])
        move(path, dest)

def get_test_labels(directory='../data/GTSRB/GT-final_test.csv'):
    annotations = []
    with open(directory) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader) # skip header
        for row in reader:
            filename = row[0]
            label = int(row[7])
            annotations.append((filename, label))
    return annotations

def clean_Nonetypes(in_data):
    out_data = []
    for d in in_data:
        d = np.array(d, dtype=np.float32)
        np.nan_to_num(d, copy=False)
        out_data.append(d)
    return out_data
