#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

def GaussianMixture_clustering(train_x, val_x, test_x, n_components = 10, covariance_type = 'tied'):
    model = GaussianMixture(n_components = n_components, covariance_type = covariance_type)

    #  train_x = dataset["train"][:][0]
    #  val_x = dataset["val"][:][0]
    #  test_x = dataset["test"][:][0]

    model.fit(train_x)
    train_pred_label = model.predict(train_x)
    model.fit(val_x)
    val_pred_label = model.predict(val_x)
    model.fit(test_x)
    test_pred_label = model.predict(test_x)

    return train_pred_label, val_pred_label, test_pred_label, model

