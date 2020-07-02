#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from .deep_embedding_clustering import dec_module


def GaussianMixture_clustering(train_x,
                               val_x,
                               test_x,
                               n_components=5,
                               covariance_type='tied'):
    model = GaussianMixture(n_components=n_components,
                            covariance_type=covariance_type)

    model.fit(train_x)
    train_pred_label = model.predict(train_x)
    model.fit(val_x)
    val_pred_label = model.predict(val_x)
    model.fit(test_x)
    test_pred_label = model.predict(test_x)

    return train_pred_label, val_pred_label, test_pred_label, model


def DeepEmbedding_clustering(train_x,
                             train_y,
                             val_x,
                             val_y,
                             test_x,
                             test_y,
                             n_components=5):
    model = dec_module(train_x, train_y, val_x, val_y, test_x, test_y,
                       n_components)
    model.fit()
    train_pred_label, val_pred_label, test_pred_label = model.predict()

    return train_pred_label, val_pred_label, test_pred_label, model
