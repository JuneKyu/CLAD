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

    # mean and std for each classes
    mean_std =  [([0.5256554097732844, 0.5603293658088235, 0.5889068098958333],
                  [0.2502201923157119, 0.24083484271296543, 0.26597347612522804]),  # class 0: airplane
                 ([0.47118413449754903, 0.4545294178921569, 0.447198645067402],
                  [0.2680635326445898, 0.26582738567068387, 0.27494592007085733]),  # class 1: automobile
                 ([0.4892499142156863, 0.49147702742034316, 0.4240447817095588],
                  [0.22705478249745528, 0.22094457725751104, 0.24337925141470124]), # class 2: bird
                 ([0.49548247012867647, 0.4564121124387255, 0.4155386358762255],
                  [0.2568431263908702, 0.25227077747941, 0.2579937168653468]),      # class 3: cat
                 ([0.47159063419117647, 0.4652057314644608, 0.3782071515012255],
                  [0.21732735231631367, 0.20652700336972213, 0.2118233405487206]),  # class 4: deer
                 ([0.4999258938419117, 0.4646367578125, 0.41654605085784313],
                  [0.25042532004474577, 0.24374875790308145, 0.2489463575086813]),  # class 5: dog
                 ([0.47005706035539213, 0.4383936764705882, 0.34521907245710787],
                  [0.22888339365158977, 0.21856169153937288, 0.22041993680516692]), # class 6: frog
                 ([0.5019583601409313, 0.479863846507353, 0.4168859995404412],
                  [0.2430489773244701, 0.24397302190495562, 0.2517155964829514]),   # class 7: horse
                 ([0.49022592524509806, 0.5253946185661764, 0.5546856449142158],
                  [0.24962469788031366, 0.24068881282532514, 0.2514975937311561]),  # class 8: ship
                 ([0.4986669837622549, 0.48534152956495097, 0.4780763526348039],
                  [0.2680525239120862, 0.2691079747712421, 0.281016526123067])]     # class 9: truck

    normal_mean = [0, 0, 0]
    normal_std = [0, 0, 0]
    for i in config.normal_class_index_list:
        for j in range(3):
            normal_mean[j] += mean_std[i][0][j]
            normal_std[j] += mean_std[i][1][j]
    for k in range(3):
        normal_mean[k] = normal_mean[k] / len(config.normal_class_index_list)
        normal_std[k] = normal_std[k] / len(config.normal_class_index_list)

    cifar10_data_path = directory
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        #  transforms.Normalize(mean=[123.3 / 255, 123.0 / 255, 113.9 / 255],
        #                       std=[63.0 / 255, 62.1 / 255, 66.7 / 255.0])
        #  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
        #                       std=[0.247, 0.243, 0.261])
        transforms.Normalize(mean=normal_mean, std=normal_std)
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
