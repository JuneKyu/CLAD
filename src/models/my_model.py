#!/usr/bin/env python3
# -*- codeing: utf-8 -*-

import numpy as np

from .cluster_models import GaussianMixture_clustering, DeepEmbedding_clustering
from .classifiers import KNN_classifier, SVM_classifier
from config import implemented_cluster_models, implemented_classifier_models

from sklearn.metrics import f1_score

from itertools import combinations

import config

import pdb

np.random.seed(777)


class Model(object):
    """
    self.train_pred_label = predicted label from clustering model
    self.val_pred_label =
    self.test_pred_label =
    self.cluster_model =
    """
    def __init__(self, dataset, cluster_num, cluster_type, classifier_type,
                 use_noise_labeling):
        """TODO: to be defined. """

        self.train_x = dataset["train"][:][0]
        self.val_x = dataset["val"][:][0]
        self.test_x = dataset["test"][:][0]
        self.train_y = dataset["train"][:][1]
        self.val_y = dataset["val"][:][1]
        self.test_y = dataset["test"][:][1]

        # cluster variables
        self.cluster_num = cluster_num
        self.cluster_type = cluster_type
        self.train_clusters = []
        self.val_clusters = []
        self.test_clusters = []
        self.cluster_model = None

        # classifier variables
        self.classifier_type = classifier_type

        #  self.use_noise_labeling = use_noise_labeling

        assert cluster_type in implemented_cluster_models
        assert classifier_type in implemented_classifier_models

    def cluster(self):
        """
        """

        #  print("clustering...")

        if self.cluster_type == 'gmm':  # gaussian mixture model
            self.train_clusters, self.val_clusters, self.test_clusters, self.cluster_model = \
                GaussianMixture_clustering(
                    train_x=self.train_x,
                    val_x=self.val_x,
                    test_x=self.test_x,
                    n_components=self.cluster_num,
                    covariance_type=config.gmm_type)

        elif self.cluster_type == 'dec':  # deep embedding clustering
            self.train_clusters, self.val_clusters, self.test_clusters, self.cluster_model = \
                DeepEmbedding_clustering(
                    train_x=self.train_x,
                    train_y=self.train_y,
                    val_x=self.val_x,
                    val_y=self.val_y,
                    test_x=self.test_x,
                    test_y=self.test_y,
                    n_components=self.cluster_num)

    def classify(self):  # classify with cluster model

        log = config.logger

        test_predict_list = []

        #  pdb.set_trace()
        # binary classification -> set one cluster as normal for each rounds of loop
        for cluster_index in range(len(np.unique(self.train_clusters))):

            # give labels for each cluster
            # -> make cluster binary; 1 for cluster_index, 0 for else
            train_cluster_labels = (
                self.train_clusters == cluster_index).astype(int)
            val_cluster_labels = (
                self.val_clusters == cluster_index).astype(int)
            test_cluster_labels = (
                self.test_clusters == cluster_index).astype(int)

            # fix when cluster index is set to '0' to all data
            if np.unique(train_cluster_labels).shape[0] == 1:
                train_cluster_labels[0] = 1
                val_cluster_labels[0] = 1
                test_cluster_labels[0] = 1

            print("classifing " + str(cluster_index) + "th cluster...")
            log.info("classifing " + str(cluster_index) + "th cluster...")

            if self.classifier_type == 'svm':
                classifier = SVM_classifier(
                    gamma=config.svm_gamma,
                    C=config.svm_C,
                    train_data=self.train_x,  # train_x
                    train_label=train_cluster_labels)  # cluster_labels

            elif self.classifier_type == 'knn':
                classifier = KNN_classifier(n_neighbors=config.knn_n_neighbors,
                                            train_data=self.train_x,
                                            train_label=train_cluster_labels)

            pred_test = classifier.predict(self.test_x)
            test_predict_list.append(pred_test)

        best_f1 = 0
        best_combi = None
        best_threshold = 0

        # find best cluster sets for every combinations
        for combi_num in range(2, self.cluster_num + 1):  # ex : 2, 3, 4, 5
            #combi_num : the number of clusters set into one

            data_combi_list = list(combinations(test_predict_list, combi_num))
            combi_list = list(
                combinations(np.arange(self.cluster_num), combi_num))

            for combi_ in range(len(data_combi_list)):
                data_predict = 0

                for i in range(len(data_combi_list[combi_])):
                    data_predict += data_combi_list[combi_][i]

                # set threshold as j (up to the number of clusters)
                #  for j in range(1, len(data_combi_list[combi_]) - 1):
                for j in range(1, len(data_combi_list[combi_])):  # TODO

                    f1 = f1_score(self.test_y, (data_predict < j).astype(int))

                    if f1 > best_f1:
                        best_f1 = f1
                        best_combi = combi_list[combi_]
                        best_threshold = j

                    print(
                        "cluster_num - {}, combination - {}, f1 - {:.4f}, threshold - {}"
                        .format(self.cluster_num, combi_list[combi_], f1, j))
                    log.info(
                        "cluster_num - {}, combination - {}, f1 - {:.4f}, threshold - {}"
                        .format(self.cluster_num, combi_list[combi_], f1, j))

        print("best f1 score : {:.4f}".format(best_f1))
        log.info("best f1 score : {:.4f}".format(best_f1))
        print("best combi = {}".format(best_combi))
        log.info("best combi = {}".format(best_combi))
        print("best_threshold : {}".format(best_threshold))
        log.info("best_threshold : {}".format(best_threshold))
