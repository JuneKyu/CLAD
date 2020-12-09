# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import SGD, Adam
import torch.nn.utils as torch_utils
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from itertools import combinations
import numpy as np
from models.utils import plot_distribution

from tqdm import tqdm

import pdb

import config

log = config.logger


def set_image_data(train_x, train_y, batch_size=32):

    num_data = train_x.shape[0]
    channel_size = train_x.shape[1]
    height = train_x.shape[2]
    width = train_x.shape[3]
    data = TensorDataset(train_x, train_y)
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader, height, width, channel_size, num_data


def set_cps_data(train_x, train_y, batch_size=32):

    num_data = train_x.shape[0]
    data = TensorDataset(train_x, train_y)
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader, num_data


def init_weights(m):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class DEC_Module():
    def __init__(self,
                 dataset_name,
                 train_x,
                 train_y,
                 batch_size,
                 cluster_type="dec",
                 n_components=5,
                 n_hidden_features=10,
                 stopping_delta=0.001):

        self.cluster_type = cluster_type
        self.stopping_delta = stopping_delta
        if (dataset_name in config.cps_datasets):
            self.dataloader, self.num_data = set_cps_data(
                train_x, train_y, batch_size=batch_size)
            self.height = 10
            self.width = 1
            self.channel_size = 1  # window size = 10 -> height * width * channel = 10
        elif (dataset_name in config.text_datasets):
            print("not implemented")
        elif (dataset_name in config.image_datasets):
            self.dataloader, self.height, self.width, self.channel_size, self.num_data = set_image_data(
                train_x, train_y, batch_size=batch_size)

        self.n_components = n_components
        self.n_hidden_features = n_hidden_features
        self.encoder = Encoder(dataset_name=dataset_name,
                               cluster_type=cluster_type,
                               height=self.height,
                               width=self.width,
                               n_components=n_components,
                               n_hidden_features=n_hidden_features).to(
                                   config.device)
        self.decoder = Decoder(dataset_name=dataset_name,
                               cluster_type=cluster_type,
                               height=self.height,
                               width=self.width,
                               n_components=n_components,
                               n_hidden_features=n_hidden_features).to(
                                   config.device)
        #  self.encoder.apply(init_weights)
        #  self.decoder.apply(init_weights)

    def plot_pretrain(self, encoder, decoder, n_components, epoch):
        encoder.eval()
        decoder.eval()
        test_x = []
        true_labels = []
        for i, d in enumerate(self.dataloader):
            out = encoder(d[0].cuda()).cpu().detach().numpy()
            test_x.extend(out)
        km = KMeans(n_clusters=n_components,
                    n_init=max(20, n_components),
                    n_jobs=-1)
        y_pred = km.fit_predict(test_x)
        if (config.plot_clustering):
            plot_distribution(
                epoch=epoch,
                train=False,
                path=config.plot_path,
                data_x=test_x,
                #  true_y=true_labels,
                pred_y=y_pred)

    def plot_train(self, dec, n_components, epoch):
        dec.eval()
        test_x = []
        true_labels = []
        for i, d in enumerate(self.dataloader):
            out = dec.module.encoder(d[0].cuda()).cpu().detach().numpy()
            test_x.extend(out)
        km = KMeans(n_clusters=n_components,
                    n_init=max(20, n_components),
                    n_jobs=-1)
        y_pred = km.fit_predict(test_x)
        if (config.plot_clustering):
            plot_distribution(
                epoch=epoch,
                train=True,
                path=config.plot_path,
                data_x=test_x,
                #  true_y=true_labels,
                pred_y=y_pred)

    def pretrain(self, epochs):

        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = nn.DataParallel(self.decoder)
        self.encoder.to(config.device)
        self.decoder.to(config.device)
        #  gradient clipping
        max_grad_norm = 3.
        torch_utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)
        torch_utils.clip_grad_norm_(self.decoder.parameters(), max_grad_norm)

        loss_function = nn.MSELoss()

        if (self.cluster_type == 'dec'):
            #  optimizer_enc = SGD(params=self.encoder.parameters(),
            #                      lr=0.01,
            #                      momentum=0.9)
            #  optimizer_dec = SGD(params=self.decoder.parameters(),
            #                      lr=0.01,
            #                      momentum=0.9)

            # testing for swat
            optimizer_enc = SGD(params=self.encoder.parameters(),
                                lr=config.dec_pretrain_lr,
                                momentum=0.9)
            optimizer_dec = SGD(params=self.decoder.parameters(),
                                lr=config.dec_pretrain_lr,
                                momentum=0.9)

        else:
            optimizer_enc = Adam(params=self.encoder.parameters())
            optimizer_dec = Adam(params=self.decoder.parameters())

        loss_value = 0

        for epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()
            data_iterator = tqdm(self.dataloader,
                                 leave='True',
                                 unit='batch',
                                 postfix={
                                     'epoch': epoch,
                                     'loss': '%.6f' % 0.0,
                                 })
            for index, batch in enumerate(data_iterator):
                if (isinstance(batch, tuple)
                        or isinstance(batch, list) and len(batch) in [1, 2]):
                    batch = batch[0]
                batch = batch.cuda(non_blocking=True)

                output = self.encoder(batch)
                output = self.decoder(output)
                loss = loss_function(output, batch)
                loss_value = float(loss.item())
                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
                loss.backward()
                optimizer_enc.step()
                optimizer_dec.step()
                data_iterator.set_postfix(
                    epoch=epoch,
                    loss='%.6f' % loss_value,
                )
        if (config.plot_clustering):
            self.plot_pretrain(self.encoder, self.decoder, self.n_components,
                               epoch)
        print("pretraining autoencoder ended.")

    def train(self, epochs):
        #  print("train epochs : {}, lr : {}, momentum : {}".format(
        #      epochs, lr, momentum))
        self.dec = DEC(self.encoder.module)
        self.dec = nn.DataParallel(self.dec)
        assert self.encoder.module == self.dec.module.encoder
        optimizer = SGD(self.dec.parameters(),
                        lr=config.dec_train_lr,
                        momentum=0.9)
        #  optimizer = Adam(params=self.dec.parameters())

        data_iterator = tqdm(self.dataloader,
                             leave='True',
                             unit='batch',
                             postfix={
                                 'epoch': -1,
                                 'acc': '%.4f' % 0.0,
                                 'loss': '%.6f' % 0.0,
                             })
        km = KMeans(n_clusters=self.n_components,
                    n_init=max(20, self.n_hidden_features),
                    n_jobs=-1)
        self.dec.train()
        self.dec.to(config.device)
        features = []
        actual = []
        for index, batch in enumerate(data_iterator):
            if ((isinstance(batch, tuple) or isinstance(batch, list))
                    and len(batch) == 2):
                batch, value = batch
                actual.append(value)
            batch = batch.cuda(non_blocking=True)
            features.append(self.dec.module.encoder(batch).detach().cpu())
        actual = torch.cat(actual).long()
        predicted = km.fit_predict(torch.cat(features).numpy())
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)

        cluster_centers = torch.tensor(km.cluster_centers_,
                                       dtype=torch.float,
                                       requires_grad=True)
        cluster_centers = cluster_centers.cuda(non_blocking=True)
        with torch.no_grad():
            self.dec.module.state_dict()['assignment.cluster_centers'].copy_(
                cluster_centers)
        loss_function = nn.KLDivLoss(size_average=False)
        delta_label = None
        for epoch in range(epochs):
            features = []
            data_iterator = tqdm(self.dataloader,
                                 leave='True',
                                 unit='batch',
                                 postfix={
                                     'epoch': epoch,
                                     'loss': '%.8f' % 0.0,
                                     'dlb': '%.4f' % (delta_label or 0.0)
                                 })
            self.dec.train()
            for index, batch in enumerate(data_iterator):
                if ((isinstance(batch, tuple) or isinstance(batch, list))
                        and len(batch) == 2):
                    batch, _ = batch
                batch = batch.cuda(non_blocking=True)
                output = self.dec(batch)
                target = target_distribution(output).detach()
                loss = loss_function(output.log(), target) / output.shape[0]
                data_iterator.set_postfix(epoch=epoch,
                                          loss='%.8f' % float(loss.item()),
                                          dlb='%.4f' % (delta_label or 0.0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
                features.append(self.dec.module.encoder(batch).detach().cpu())
                if index % 10 == 0:  # update_freq = 10
                    loss_value = float(loss.item())
                    data_iterator.set_postfix(
                        epoch=epoch,
                        loss='%.8f' % loss_value,
                        dlb='%.4f' % (delta_label or 0.0),
                    )
            predicted, actual = self.predict()
            delta_label = float(
                (predicted != predicted_previous
                 ).float().sum().item()) / predicted_previous.shape[0]
            if self.stopping_delta is not None and delta_label < self.stopping_delta:
                print(
                    'Early stopping as label delta "%1.5f" less tahn "%1.5f".'
                    % (delta_label, self.stopping_delta))
                break
            predicted_previous = predicted

        if (config.plot_clustering):
            self.plot_train(self.dec, self.n_components, epoch)
        self.encoder = self.dec.module.encoder
        print("training dec ended.")

    def predict(self):
        features = []
        actual = []
        self.dec.eval()
        #  for batch in data_iterator:
        for batch in self.dataloader:
            if ((isinstance(batch, tuple) or isinstance(batch, list))
                    and len(batch) == 2):
                batch, value = batch
                actual.append(value)
            batch = batch.cuda(non_blocking=True)
            features.append(self.dec(batch).detach().cpu())
        return torch.cat(features).max(1)[1], torch.cat(actual).long()


class DEC(nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.assignment = ClusterAssignment(
            self.encoder.n_components,
            self.encoder.n_hidden_features,
            alpha=1.0
        )  # alpha represent the degrees of freedom in the t-distribution

    def forward(self, x):
        out = self.assignment(self.encoder(x))
        return out

    def predict(self, x):
        with torch.no_grad():
            out = self.assignment(self.encoder(x))
        return out

    def encode(self, x):
        with torch.no_grad():
            out = self.encoder(x)
        return out


class ClusterAssignment(nn.Module):
    def __init__(self,
                 cluster_number: int,
                 embedding_dimension: int,
                 alpha: float = 1.0,
                 cluster_centers: Optional[torch.Tensor] = None) -> None:
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number,
                                                  self.embedding_dimension,
                                                  dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    #  compute the soft assignment for a batch of feature vectors, returning a batch of assignments
    #  for each cluster.
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum(
            (batch.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class Encoder(nn.Module):
    def __init__(self, dataset_name, cluster_type, height, width, n_components,
                 n_hidden_features):
        super(Encoder, self).__init__()
        self.dataset_name = dataset_name
        self.height = height
        self.width = width
        self.cluster_type = cluster_type
        self.n_components = n_components
        self.n_hidden_features = n_hidden_features
        self.channels = config.cvae_channel
        self.ksize = config.cvae_kernel_size

        if (cluster_type == 'dec'):
            self.dropout = nn.Dropout(p=0.1)
            self.encoder_net = nn.Sequential(
                nn.Linear(self.channels * height * width, 500), nn.ReLU(),
                nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000),
                nn.ReLU(), nn.Linear(2000, n_hidden_features))

        elif (cluster_type == 'cvae_base'):
            self.dropout = nn.Dropout(p=0.1)
            self.encoder_net = nn.Sequential(
                nn.Conv2d(in_channels=self.channels,
                          out_channels=self.channels * 4,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                #  nn.ReLU(),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=self.channels * 4,
                          out_channels=self.channels * 8,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                #  nn.ReLU(),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=self.channels * 8,
                          out_channels=self.channels * 16,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                #  nn.ReLU(),
                nn.ELU(),
                Flatten(),
                nn.Linear((self.height // (2**2)) * (self.width // (2**2)) *
                          self.channels * 16, 512),
                nn.ReLU(),
                #  nn.ELU(),
                nn.Linear(512, self.n_hidden_features * 2))
        elif (cluster_type == 'cvae_large'):
            self.dropout = nn.Dropout(p=0.1)
            self.encoder_net = nn.Sequential(
                nn.Conv2d(in_channels=self.channels,
                          out_channels=self.channels * 16,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 16),
                nn.ELU(),
                nn.Conv2d(in_channels=self.channels * 16,
                          out_channels=self.channels * 16,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 16),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=self.channels * 16,
                          out_channels=self.channels * 32,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 32),
                nn.ELU(),
                nn.Conv2d(in_channels=self.channels * 32,
                          out_channels=self.channels * 32,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 32),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=self.channels * 32,
                          out_channels=self.channels * 64,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 64),
                nn.ELU(),
                nn.Conv2d(in_channels=self.channels * 64,
                          out_channels=self.channels * 64,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 64),
                nn.ELU(),
                Flatten(),
                nn.Linear((self.height // (2**2)) * (self.width // (2**2)) *
                          self.channels * 64, 512),
                #  nn.ELU(),
                nn.ReLU(),
                nn.Linear(512, self.n_hidden_features * 2))
        elif (cluster_type == 'cvae_temp'):
            self.dropout = nn.Dropout(p=0.1)
            self.encoder_net = nn.Sequential(
                nn.Conv2d(in_channels=self.channels,
                          out_channels=self.channels * 32,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(in_channels=self.channels * 32,
                          out_channels=self.channels * 64,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(in_channels=self.channels * 64,
                          out_channels=self.channels * 128,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(in_channels=self.channels * 128,
                          out_channels=self.channels * 256,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2), nn.ReLU(), Flatten(),
                nn.Linear((self.height // (2**3)) * (self.width // (2**3)) *
                          self.channels * 256, 512), nn.ReLU(),
                nn.Linear(512, self.n_hidden_features * 2))

    def split_z(self, z):
        z_mu = z[:, :self.n_hidden_features]
        z_sigma = z[:, self.n_hidden_features:]
        return z_mu, z_sigma

    def sample_z(self, mu, sigma):
        epsilon = torch.randn_like(mu)
        sample = mu + (sigma * epsilon)
        return sample

    def forward(self, x):
        if (self.cluster_type == 'dec'):
            x = x.reshape(-1, self.channels * self.height * self.width)
        out = self.dropout(x)
        out = self.encoder_net(out)
        if (self.cluster_type != 'dec'):
            z_mu, z_sigma = self.split_z(z=out)
            out = self.sample_z(mu=z_mu, sigma=z_sigma)
        return out

    def predict(self, x):
        if (self.cluster_type == 'dec'):
            x = x.reshape(-1, self.channels * self.heigt * self.width)
        with torch.no_grad():
            out = self.encoder_net(out)
            z_mu, z_sigma = self.split_z(z=out),
            out = self.sample_z(mu=z_mu, sigma=z_sigma)
        return out


class Decoder(nn.Module):
    def __init__(self, dataset_name, cluster_type, height, width, n_components,
                 n_hidden_features):
        super(Decoder, self).__init__()
        self.dataset_name = dataset_name
        self.height = height
        self.width = width
        self.cluster_type = cluster_type
        self.n_components = n_components
        self.n_hidden_features = n_hidden_features
        self.channels = config.cvae_channel
        self.ksize = config.cvae_kernel_size

        if (cluster_type == 'dec'):
            self.decoder_net = nn.Sequential(
                nn.Linear(n_hidden_features, 2000), nn.ReLU(),
                nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500),
                nn.ReLU(), nn.Linear(500, self.channels * height * width))

        #  elif (cluster_type == 'cvae_small'):

        elif (cluster_type == 'cvae_base'):
            self.decoder_dense = nn.Sequential(
                nn.Linear(self.n_hidden_features, 512),
                nn.ReLU(),
                #  nn.ELU(),
                nn.Linear(512, (self.height // (2**2)) *
                          (self.width // (2**2)) * self.channels * 16),
                nn.ReLU(),
                #  nn.ELU(),
            )
            self.decoder_net = nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.channels * 16,
                                   out_channels=self.channels * 8,
                                   kernel_size=self.ksize + 1,
                                   stride=2,
                                   padding=1),
                #  nn.ReLU(),
                nn.ELU(),
                nn.ConvTranspose2d(in_channels=self.channels * 8,
                                   out_channels=self.channels * 4,
                                   kernel_size=self.ksize + 1,
                                   stride=2,
                                   padding=1),
                #  nn.ReLU(),
                nn.ELU(),
                nn.Conv2d(in_channels=self.channels * 4,
                          out_channels=self.channels,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.Sigmoid(),
            )
        elif (cluster_type == 'cvae_large'):
            self.decoder_dense = nn.Sequential(
                #  nn.Linear(self.n_hidden_features, (self.height // (2**2)) *
                #            (self.width // (2**2)) * self.channels * 64),
                #  nn.ELU(),
                nn.Linear(self.n_hidden_features, 512),
                #  nn.ELU(),
                nn.ReLU(),
                nn.Linear(512, (self.height // (2**2)) *
                          (self.width // (2**2)) * self.channels * 64),
                #  nn.ELU(),
                nn.ReLU(),
            )
            self.decoder_net = nn.Sequential(
                #  nn.BatchNorm2d(self.channels * 64),
                nn.Conv2d(in_channels=self.channels * 64,
                          out_channels=self.channels * 64,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 64),
                nn.ELU(),
                nn.ConvTranspose2d(in_channels=self.channels * 64,
                                   out_channels=self.channels * 32,
                                   kernel_size=self.ksize + 1,
                                   stride=2,
                                   padding=1),
                nn.BatchNorm2d(self.channels * 32),
                nn.ELU(),
                nn.Conv2d(in_channels=self.channels * 32,
                          out_channels=self.channels * 32,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 32),
                nn.ELU(),
                nn.ConvTranspose2d(in_channels=self.channels * 32,
                                   out_channels=self.channels * 16,
                                   kernel_size=self.ksize + 1,
                                   stride=2,
                                   padding=1),
                nn.BatchNorm2d(self.channels * 16),
                nn.ELU(),
                nn.Conv2d(in_channels=self.channels * 16,
                          out_channels=self.channels * 16,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.BatchNorm2d(self.channels * 16),
                nn.ELU(),
                nn.Conv2d(in_channels=self.channels * 16,
                          out_channels=self.channels,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.Sigmoid(),
            )
        elif (cluster_type == 'cvae_temp'):
            self.decoder_dense = nn.Sequential(
                nn.Linear(self.n_hidden_features, 512),
                nn.ReLU(),
                nn.Linear(512, (self.height // (2**3)) *
                          (self.width // (2**3)) * self.channels * 256),
                nn.ReLU(),
            )
            self.decoder_net = nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.channels * 256,
                                   out_channels=self.channels * 128,
                                   kernel_size=self.ksize + 1,
                                   stride=2,
                                   padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=self.channels * 128,
                                   out_channels=self.channels * 64,
                                   kernel_size=self.ksize + 1,
                                   stride=2,
                                   padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=self.channels * 64,
                                   out_channels=self.channels * 32,
                                   kernel_size=self.ksize + 1,
                                   stride=2,
                                   padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.channels * 32,
                          out_channels=self.channels,
                          kernel_size=self.ksize,
                          stride=1,
                          padding=self.ksize // 2),
                nn.Sigmoid(),
            )

    def forward(self, x):
        if (self.cluster_type == 'dec'):
            out = self.decoder_net(x)
        elif (self.cluster_type == 'cvae_base'):
            out = self.decoder_dense(x)
            out = out.reshape(out.size(0), self.channels * 16,
                              (self.height // (2**2)), (self.width // (2**2)))
            out = self.decoder_net(out)
        elif (self.cluster_type == 'cvae_large'):
            out = self.decoder_dense(x)
            out = out.reshape(out.size(0), self.channels * 64,
                              (self.height // (2**2)), (self.width // (2**2)))
            out = self.decoder_net(out)
        elif (self.cluster_type == 'cvae_temp'):
            out = self.decoder_dense(x)
            out = out.reshape(out.size(0), self.channels * 256,
                              (self.height // (2**3)), (self.width // (2**3)))
            out = self.decoder_net(out)
        if (self.dataset_name in config.cps_datasets):
            # 10 is the size of unit of freq_data from swat
            # 1280 -> 128*10
            out = out.reshape(-1, 10)
        elif (self.dataset_name in config.text_datasets):
            print("not implemented")
        elif (self.dataset_name in config.image_datasets):
            out = out.reshape(-1, self.channels, self.height, self.width)
        return out

    def predict(self, x):
        with torch.nograd():
            if (self.cluster_type == 'dec'):
                out = self.decoder_net(x)
            elif (self.cluster_type == 'cvae_base'):
                out = self.decoder_dense(x)
                out = out.reshape(out.size(0), self.channels * 16,
                                  (self.height // (2**2)),
                                  (self.width // (2**2)))
                out = self.decoder_net(out)
            elif (self.cluster_type == 'cvae_large'):
                out = self.decoder_dense(x)
                out = out.reshape(out.size(0), self.channels * 64,
                                  (self.height // (2**2)),
                                  (self.width // (2**2)))
                out = self.decoder_net(out)
            elif (self.cluter_type == 'cvae_temp'):
                out = self.decoder_dense(x)
                out = out.reshape(out.size(0), self.channels * 256,
                                  (self.height // (2**3)),
                                  (self.width // (2**3)))
        out = out.reshape(-1, self.channels, self.height, self.width)
        return out


def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch**2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


# check all the possible combinations disregarding num of normal class list
def binary_cluster_accuracy(y_true,
                            y_predicted,
                            cluster_number: Optional[int] = None):
    if cluster_number is None:
        cluster_number = max(y_predicted.max(),
                             y_true.max()) + 1  # assume labels are 0-indexed
    cluster_indexes = [i for i in range(cluster_number)]
    best_acc = 0.0
    best_combi = []
    reassigned_ind = []
    for num_of_normal_indexes in cluster_indexes:
        if (num_of_normal_indexes == 0 or num_of_normal_indexes == 1): continue
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
            if (acc > best_acc):
                best_acc = acc
                best_combi = combination
                reassigned_ind = pred

    print("best_acc : {} , best_combi : {}".format(best_acc, best_combi))
    return reassigned_ind, best_acc, best_combi
