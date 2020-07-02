# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from ptsdae.sdae import StackedDenoisingAutoEncoder

from ptdec.dec import DEC
from ptdec.model import train, predict
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy

import pdb

import config


class dec_module():
    def __init__(self,
                 train_x,
                 train_y,
                 val_x,
                 val_y,
                 test_x,
                 test_y,
                 n_components=5):
        self.cuda = False
        self.input_dim = len(train_x[0])
        self.ds_train = CachedData(data_x=train_x,
                                   data_y=train_y,
                                   cuda=self.cuda)
        self.ds_val = CachedData(data_x=val_x, data_y=val_y, cuda=self.cuda)
        self.ds_test = CachedData(data_x=test_x, data_y=test_y, cuda=self.cuda)
        self.n_components = n_components

    def fit(self):
        autoencoder = StackedDenoisingAutoEncoder(
            [self.input_dim, 500, 500, 2000, self.n_components],
            final_activation=None)
        #  autoencoder = StackedDenoisingAutoEncoder([28 * 28, 500, 500, 2000, 10], final_activation=None)

        if self.cuda:
            autoencoder.cuda()
        print('Pretraining stage.')
        pdb.set_trace()
        if config.set_dec_lower_learning_rate:
            ae.pretrain(
                self.ds_train,
                autoencoder,
                cuda=self.cuda,
                validation=self.ds_val,
                epochs=config.dec_pretrain_epochs,
                batch_size=config.dec_batch_size,
                optimizer=lambda model: SGD(model.parameters(),
                                            lr=config.lower_lr,
                                            momentum=config.lower_momentum),
                scheduler=lambda x: StepLR(x, 100, gamma=0.1),
                corruption=0.2)
        else:
            ae.pretrain(self.ds_train,
                        autoencoder,
                        cuda=self.cuda,
                        validation=self.ds_val,
                        epochs=config.dec_pretrain_epochs,
                        batch_size=config.dec_batch_size,
                        optimizer=lambda model: SGD(
                            model.parameters(), lr=0.1, momentum=0.9),
                        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
                        corruption=0.2)
        print('Training stage.')
        ae_optimizer = SGD(params=autoencoder.parameters(),
                           lr=0.1,
                           momentum=0.9)
        ae.train(self.ds_train,
                 autoencoder,
                 cuda=self.cuda,
                 validation=self.ds_val,
                 epochs=config.dec_finetune_epochs,
                 batch_size=config.dec_batch_size,
                 optimizer=ae_optimizer,
                 scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
                 corruption=0.2)
        print('DEC stage.')
        self.model = DEC(cluster_number=self.n_components,
                         hidden_dimension=self.n_components,
                         encoder=autoencoder.encoder)
        if self.cuda:
            self.model.cuda()
        dec_optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        train(
            dataset=self.ds_train,
            model=self.model,
            epochs=100,  # 100
            batch_size=256,
            optimizer=dec_optimizer,
            stopping_delta=0.000001,
            cuda=self.cuda)

    def predict(self):
        train_predicted, train_actual = predict(self.ds_train,
                                                self.model,
                                                1024,
                                                silent=True,
                                                return_actual=True,
                                                cuda=self.cuda)
        train_predicted = train_predicted.cpu().numpy()
        train_actual = train_actual.cpu().numpy()
        _, train_accuracy = cluster_accuracy(train_actual, train_predicted)
        log = config.logger
        print("DEC training accuracy : {}".format(train_accuracy))
        log.info("DEC training accuracy : {}".format(train_accuracy))
        # prediction variable : shape = torch.size(6023)
        #                     : type = torch.Tensor
        val_predicted, _ = predict(self.ds_val,
                                   self.model,
                                   1024,
                                   silent=True,
                                   return_actual=True,
                                   cuda=self.cuda)
        #  val_actual = val_actual.cpu().numpy()
        val_predicted = val_predicted.cpu().numpy()
        test_predicted, _ = predict(self.ds_test,
                                    self.model,
                                    1024,
                                    silent=True,
                                    return_actual=True,
                                    cuda=self.cuda)
        #  test_actual = test_actual.cpu().numpy()
        test_predicted = test_predicted.cpu().numpy()

        return train_predicted, val_predicted, test_predicted


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
