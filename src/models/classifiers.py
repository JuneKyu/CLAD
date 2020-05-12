#!/usr/bin/env python3
# -*- codeing: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# KNN
def KNN_classifier(n_neighbors, train_data, train_label):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(train_data, train_label)
    return model

# svm
def SVM_classifier(gamma, C, train_data, train_label):
    model = svm.SVC(gamma=gamma, C=C)
    model.fit(train_data, train_label)
    return model


