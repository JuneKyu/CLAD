#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import config


def plot_distribution(epoch,
                      train,
                      acc,
                      path,
                      data_x,
                      true_y,
                      pred_y,
                      learning_rate=100,
                      n_jobs=-1):
    print("plotting image on " + path + "...")
    if (os.path.exists(path) == False):
        os.makedirs(path)
    tsne_model = TSNE(n_components=2,
                      learning_rate=learning_rate,
                      n_jobs=n_jobs)
    #  pca_model = PCA(n_components=2)

    data_x = np.array(data_x)
    if (len(data_x.shape) > 2):
        data_temp = []
        for data in data_x:
            data_temp.append(data.rehsape(-1))
        data_x = np.array(data_temp)

    transformed = tsne_model.fit_transform(data_x)
    #  transformed = pca_model.fit_transform(data_x)
    xs = transformed[:, 0]
    ys = transformed[:, 1]

    draw_plot(xs, ys, train, epoch, true_y, acc,
              os.path.join(path, "true_label"))
    draw_plot(xs, ys, train, epoch, pred_y, acc,
              os.path.join(path, "pred_label"))


def draw_plot(xs, ys, train, epoch, label, acc, path):
    if (os.path.exists(path) == False):
        os.makedirs(path)
    plt.scatter(xs, ys, c=label)
    plt.title('acc: ' + str(acc), loc='center')
    plt.grid()
    plt.legend(np.unique(label))
    if (epoch == -1):
        epoch = "_init_"
    else:
        epoch = str(epoch)
    if (train):
        plt.savefig(os.path.join(path, 'fig' + epoch + '_train.png'), dpi=300)
    else:
        plt.savefig(os.path.join(path, 'fig' + epoch + '_pretrain.png'),
                    dpi=300)
