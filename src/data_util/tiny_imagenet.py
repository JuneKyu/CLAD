import os
import wget
import pickle
import zipfile
import glob
from shutil import move, copytree, rmtree
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import TensorDataset

from data_util.utils import divide_data_label

import config
import pdb

#  from torchvision.datasets import

class TINY_Imagenet_Dataset(object):
    """Docstring for TINY_Imagenet_Dataset. """
    def __init__(self, root_dir: str):

        self.train, self.test = tiny_imagenet_dataset(root_dir)
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


def tiny_imagenet_dataset(directory='../data'):

    if (os.path.exists(os.path.join(directory, 'tiny-imagenet-200')) == False):
        tiny_imagenet_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        wget.download(tiny_imagenet_url, out=directory)
        tiny_imagenet_zip = zipfile.ZipFile(os.path.join(directory, 'tiny-imagenet-200.zip'))
        tiny_imagenet_zip.extractall(directory)
        os.remove(os.path.join(directory, 'tiny-imagenet-200.zip'))
        print("labeling test dataset")
        test_dataset_labeling(os.path.join(directory, 'tiny-imagenet-200/val'))

    train_path = os.path.join(directory, 'tiny-imagenet-200/train')
    test_path = os.path.join(directory, 'tiny-imagenet-200/val')

    train_classes = []
    with open(os.path.join(directory, 'tiny-imagenet-200/tiny_imagenet_classes.txt'), 'r') as f:
        item = f.read()
        train_classes = item.split('\n')[:-1]

    animal = [66, 90, 134, 139, 148, 180, 182, 191]
    insect = [13, 31, 92, 123, 164, 177, 196, 199]
    instruments = [16, 17, 18, 72, 74, 116, 128, 197]
    structure = [48, 58, 69, 96, 122, 151, 157, 178]
    transportation = [0, 22, 23, 26, 46, 47, 156, 169]
    total_index = animal + insect + instruments + structure + transportation

    animal = change_index(train_classes, animal)
    insect = change_index(train_classes, insect)
    instruments = change_index(train_classes, instruments)
    structure = change_index(train_classes, structure)
    transportation = change_index(train_classes, transportation)

    total = animal + insect + instruments + structure + transportation
    scenario_classes = (animal, insect, instruments, structure, transportation)
    total.sort()

    normal_scenario = scenario_classes[config.normal_class_index_list[0]]

    #  config.normal_class_index_list
    normal_class_index_list = []
    for normal in normal_scenario:
        normal_class_index_list.append(total.index(normal))
    config.normal_class_index_list = normal_class_index_list

    mean_std_pickle_path = 'tiny_imagenet_200_mean_std.pkl'
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

    tiny_imagenet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
        #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    selected_train_path = select_from_data(train_path, total)
    selected_test_path = select_from_data(test_path, total)
    
    train = datasets.ImageFolder(selected_train_path, transform=tiny_imagenet_transform)
    test = datasets.ImageFolder(selected_test_path, transform=tiny_imagenet_transform)

    return train, test


def select_from_data(data_path, selected_list):
    
    selected_path = os.path.join(data_path, 'selected')
    if (os.path.exists(selected_path)):
        rmtree(selected_path)

    paths = glob.glob(os.path.join(data_path, '*'))
    os.makedirs(selected_path)
    for path in paths:
        file = path.split('/')[-1]
        if (file in selected_list):
            dest = os.path.join(selected_path, file)
            copytree(path, dest)

    return selected_path


def test_dataset_labeling(test_path):
    val_dict = {}
    with open(os.path.join(test_path, 'val_annotations.txt'), 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob(os.path.join(test_path, 'images/*'))
    #  paths[0].split('/')[-1]
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(os.path.join(test_path, str(folder))):
            os.mkdir(os.path.join(test_path, str(folder)))

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = os.path.join(test_path, str(folder), str(file))
        move(path, dest)

    os.remove(os.path.join(test_path, 'val_annotations.txt'))
    os.rmdir(os.path.join(test_path, 'images'))


def change_index(classes, indexes):

    class_indexes = []
    for i in indexes:
        class_indexes.append(classes[i])

    return class_indexes
