# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.nn.utils as torch_utils
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from ptsdae.sdae import StackedDenoisingAutoEncoder

from ptdec.dec import DEC
from ptdec.model import train, predict
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy

from typing import Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import combinations
import numpy as np

import pdb

import config


class dec_module():
    def __init__(self, train_x, train_y, n_components=5):
        self.cuda = config.device
        self.input_dim = len(train_x[0])
        self.ds_train = CachedData(data_x=train_x,
                                   data_y=train_y,
                                   cuda=self.cuda)
        #  self.ds_val = CachedData(data_x=val_x, data_y=val_y, cuda=self.cuda)
        #  self.ds_test = CachedData(data_x=test_x, data_y=test_y, cuda=self.cuda)
        self.n_components = n_components

    def fit(self):
        autoencoder = StackedDenoisingAutoEncoder(
            [self.input_dim, 500, 500, 2000, self.n_components],
            final_activation=None)
        #  autoencoder = StackedDenoisingAutoEncoder([28 * 28, 500, 500, 2000, 10], final_activation=None)
        max_grad_norm = 5.
        torch_utils.clip_grad_norm_(autoencoder.parameters(), max_grad_norm)

        if self.cuda:
            autoencoder.cuda()
        print('Pretraining stage.')
        ae.pretrain(
            self.ds_train,
            autoencoder,
            cuda=self.cuda,
            #  validation=self.ds_val,
            epochs=config.dec_pretrain_epochs,
            batch_size=config.dec_batch_size,
            optimizer=lambda model: SGD(model.parameters(),
                                        lr=config.dec_pretrain_lr,
                                        momentum=config.dec_pretrain_momentum),
            scheduler=lambda x: StepLR(x, 100, gamma=0.1),
            corruption=0.2)
        print('Training stage.')
        ae_optimizer = SGD(params=autoencoder.parameters(),
                           lr=config.dec_finetune_lr,
                           momentum=config.dec_finetune_momentum)
        ae.train(
            self.ds_train,
            autoencoder,
            cuda=self.cuda,
            #  validation=self.ds_val,
            epochs=config.dec_finetune_epochs,
            batch_size=config.dec_batch_size,
            optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer,
                             config.dec_finetune_decay_step,
                             gamma=config.dec_finetune_decay_rate),
            corruption=0.2)

        print('DEC stage.')
        self.model = DEC(cluster_number=self.n_components,
                         hidden_dimension=self.n_components,
                         encoder=autoencoder.encoder)
        if self.cuda:
            self.model.cuda()
        dec_optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        train(dataset=self.ds_train,
              model=self.model,
              epochs=config.dec_train_epoch,
              batch_size=256,
              optimizer=dec_optimizer,
              stopping_delta=0.000001,
              cuda=self.cuda)

    def predict(self):
        log = config.logger

        train_predicted, train_actual = predict(self.ds_train,
                                                self.model,
                                                1024,
                                                silent=True,
                                                return_actual=True,
                                                cuda=self.cuda)
        train_predicted = train_predicted.cpu().numpy()
        train_actual = train_actual.cpu().numpy()

        print("finding the best combination for the clusters")
        train_predicted, train_accuracy, best_combi = binary_cluster_accuracy(
            train_actual, train_predicted)

        train_predicted = np.array(train_predicted)

        #
        #
        #
        #
        #
        #
        #

        # for dec clustering acc, auc
        #  train_predicted, train_accuracy, train_normal_clusters, train_auc, train_normal_clusters_auc = binary_cluster_accuracy(
        #      train_actual, train_predicted)

        #  print("DEC training accuracy : {}".format(train_accuracy))
        #  log.info("DEC training accuracy : {}".format(train_accuracy))
        #  print("DEC normal_cluster_indexes : {}".format(train_normal_clusters))
        #  log.info(
        #      "DEC normal_cluster_indexes : {}".format(train_normal_clusters))
        #  print("DEC training auc : {}".format(train_auc))
        #  log.info("DEC training auc : {}".format(train_auc))
        #  print("DEC normal cluster indexes auc : {}".format(
        #      train_normal_clusters_auc))
        #  log.info("DEC normal cluster indexes auc : {}".format(
        #      train_normal_clusters_auc))

        #  pdb.set_trace()

        #  val_predicted, val_actual = predict(self.ds_val,
        #                                      self.model,
        #                                      1024,
        #                                      silent=True,
        #                                      return_actual=True,
        #                                      cuda=self.cuda)
        #  val_predicted = val_predicted.cpu().numpy()
        #  val_actual = val_actual.cpu().numpy()
        #  val_predicted, val_accuracy = binary_cluster_accuracy(
        #  val_actual, val_predicted)

        #  print("DEC validataion accuracy : {}".format(val_accuracy))
        #  log.info("DEC validation accuracy : {}".format(val_accuracy))

        #  test_predicted, test_actual = predict(self.ds_test,
        #                                        self.model,
        #                                        1024,
        #                                        silent=True,
        #                                        return_actual=True,
        #                                        cuda=self.cuda)
        #  test_predicted = test_predicted.cpu().numpy()
        #  test_actual = test_actual.cpu().numpy()
        #  test_predicted, test_accuracy = binary_cluster_accuracy(
        #  test_actual, test_predicted)

        #  print("DEC testing accuracy : {}".format(test_accuracy))
        #  log.info("DEC testing accuracy : {}".format(test_accuracy))

        return train_predicted


def pred_labels_to_binary_labels(predicted, reassignment):
    for i in range(len(predicted)):
        if reassignment[predicted[i]] in config.normal_class_index_list:
            predicted[i] = 0
        else:
            predicted[i] = 1
    return predicted


def actual_labels_to_binary_labels(actual):
    for i in range(len(actual)):
        if actual[i] in config.normal_class_index_list:
            actual[i] = 0
        else:
            actual[i] = 1
    return actual


# check only the set of with num of normal class list
#  def binary_cluster_accuracy(y_true,
#                              y_predicted,
#                              cluster_number: Optional[int] = None):
#      if cluster_number is None:
#          cluster_number = max(y_predicted.max(),
#                               y_true.max()) + 1  # assume labels are 0-indexed
#
#      cluster_indexes = [i for i in range(cluster_number)]
#      num_of_normal_indexes = len(config.normal_class_index_list)
#      combination_list = combinations(cluster_indexes, num_of_normal_indexes)
#
#      best_acc = 0.0
#      best_combi = []
#      reassigned_ind = []
#      for combination in combination_list:
#          # for each combination
#          pred = []
#          for i in range(len(y_true)):
#              if (y_predicted[i] in combination):
#                  pred.append(0)
#              else:
#                  pred.append(1)
#          acc = accuracy_score(y_true, pred)
#          if (acc > best_acc):
#              best_acc = acc
#              best_combi = combination
#              reassigned_ind = pred
#      return reassigned_ind, best_acc


# check all the possible combinations disregarding num of normal class list
def binary_cluster_accuracy(y_true,
                            y_predicted,
                            cluster_number: Optional[int] = None):
    if cluster_number is None:
        cluster_number = max(y_predicted.max(),
                             y_true.max()) + 1  # assume labels are 0-indexed

    cluster_indexes = [i for i in range(cluster_number)]
    #  num_of_normal_indexes = len(config.normal_class_index_list)

    best_acc = 0.0
    best_combi = []
    reassigned_ind = []
    #  best_auc = 0.0
    #  best_combi_auc = []

    for num_of_normal_indexes in cluster_indexes:
        combination_list = combinations(cluster_indexes, num_of_normal_indexes)
        for combination in combination_list:
            # for each combination
            pred = []
            for i in range(len(y_true)):
                if (y_predicted[i] in combination):
                    pred.append(0)
                else:
                    pred.append(1)
            acc = accuracy_score(y_true, pred)
            #  auc = roc_auc_score(y_true, pred)
            if (acc > best_acc):
                best_acc = acc
                best_combi = combination
                reassigned_ind = pred
            #  if (auc > best_auc):
            #      best_auc = auc
            #      best_combi_auc = combination

    #  print("best_acc : {} , best_combi : {}".format(best_acc, best_combi))
    #  print("best_auc : {} , best_combi : {}".format(best_auc, best_combi_auc))

    #  return reassigned_ind, best_acc, best_combi, best_auc, best_combi_auc
    return reassigned_ind, best_acc, best_combi


class CachedData(Dataset):
    def __init__(self, data_x, data_y, cuda, testing_mode=False):

        if not cuda:
            data_x.detach().cpu()
            data_y.detach().cpu()
        self.ds = TensorDataset(data_x, data_y)
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(
                    non_blocking=True)
                self._cache[index][1] = self._cache[index][1].cuda(
                    non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)
