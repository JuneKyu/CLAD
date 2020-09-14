from .cluster_models import GaussianMixture_clustering, DeepEmbedding_clustering
from .classifiers import KNN_classifier, SVM_classifier, Linear_classifier, FC3_classifier, CNN_classifier
from config import implemented_cluster_models, implemented_classifier_models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import combinations

# temporary import to test mnist data normalization
import torch
import torchvision
from torchvision import transforms
from data_util.utils import divide_data_label

from .odin import apply_odin
from .metric import calculate_metric

import os
import numpy as np
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
    def __init__(self, dataset_name, dataset, cluster_num, cluster_type,
                 classifier_type):
        """TODO: to be defined. """

        self.dataset_name = dataset_name
        self.dec_train = dataset["dec_train"]
        self.dec_train_y = dataset["dec_train_y"]
        self.train = dataset["train"]
        self.test_in = dataset["test_in"]
        self.test_out = dataset["test_out"]

        # cluster variables
        self.cluster_num = cluster_num
        self.cluster_type = cluster_type
        self.train_clusters = []
        self.cluster_model = None

        # classifier variables
        self.classifier_type = classifier_type

        assert cluster_type in implemented_cluster_models
        assert classifier_type in implemented_classifier_models

    def cluster(self):
        """
        clustering module of the model
        """

        if self.cluster_type == 'gmm':  # gaussian mixture model
            print("not modified to new code")
            #  self.train_clusters, self.val_clusters, self.test_clusters, self.cluster_model = \
            #  GaussianMixture_clustering(
            #      train_x=self.train_x,
            #      val_x=self.val_x,
            #      test_x=self.test_x,
            #      n_components=self.cluster_num,
            #      covariance_type=config.gmm_type)

        elif self.cluster_type == 'dec' or 'cvae':  # deep embedding clustering
            if (self.dataset_name == 'mnist'):
                # use dec default configuration
                print("")
            elif (self.dataset_name == 'cifar10'):
                config.dec_pretrain_epochs = config.cifar10_dec_pretrain_epochs
                config.dec_finetune_epochs = config.cifar10_dec_finetune_epochs
                config.dec_finetune_lr = config.cifar10_dec_finetune_lr
                config.dec_finetune_momentum = config.cifar10_dec_finetune_momentum
                config.dec_finetune_decay_step = config.cifar10_dec_finetune_decay_step
                config.dec_finetune_decay_rate = config.cifar10_dec_finetune_decay_rate
                config.dec_train_epochs = config.cifar10_dec_train_epochs
                config.dec_train_lr = config.cifar10_dec_train_lr
            elif (self.dataset_name == 'reuters'):
                config.dec_finetune_epochs = config.reuters_dec_finetune_epochs
                config.dec_finetune_lr = config.reuters_dec_finetune_lr
                config.dec_finetune_decay_step = config.reuters_dec_finetune_decay_step
                config.dec_finetune_decay_rate = config.reuters_dec_finetune_decay_rate
                config.dec_train_epochs = config.reuters_dec_train_epochs

            #  pdb.set_trace()
            self.train_clusters, self.cluster_model = \
                DeepEmbedding_clustering(
                    train_x=self.dec_train,
                    train_y=self.dec_train_y, # use label only for checking the score
                    n_components=self.cluster_num)

            #  if os.path.exists(config.temp_dec_cluster):
            #      #  print("use preprocessed dec specs for mnist")
            #      self.train_clusters = np.load(
            #          os.path.join(config.temp_dec_cluster,
            #                       "train_clusters.npy"))
            #  else:
            #      os.makedirs(config.temp_dec_cluster)
            #      self.train_clusters, self.cluster_model = \
            #          DeepEmbedding_clustering(
            #              train_x=self.dec_train,
            #              train_y=self.dec_train_y, # need label only for checking the score
            #              n_components=self.cluster_num)
            #      np.save(
            #          os.path.join(config.temp_dec_cluster, "train_clusters"),
            #          self.train_clusters)

        #  elif self.cluster_type == 'cvae':  # convolutional variational auto encoder
        #      print("")

    # classify with cluster models using neural network models
    def classify_nn(self, dataset_name):

        log = config.logger
        if dataset_name in config.cps_datasets:
            print("not implemented")
        elif dataset_name in config.text_datasets:
            print("")
            # normalize mnist dataset
            #  transforms.Normalize((0.1307,), (0.3081,)
            classifier = Linear_classifier(self.train_x,
                                           self.train_clusters,
                                           n_epochs=5000,
                                           lr=0.001)
            train_pred = classifier.predict(self.train_x.cuda(config.device))
            train_accuracy = accuracy_score(train_pred, self.train_clusters)

            print(
                "NN Classifier training accuracu : {}".format(train_accuracy))
            log.info(
                "NN Classifier training accuracy = {}".format(train_accuracy))
            apply_odin(classifier, self.test_in, self.test_out)
            calculate_metric("mnist")
            #  calculate_metric("reuters")

            #  train_pred = classifier(self.)
            #  classifier = WideResNet_classifier()
            #  classifier = GRU_text_classifier(self.train_x, self.train_clusters,
            #  self.test_x, self.test_clusters)
        elif dataset_name in config.image_datasets:

            classifier_name = config.classifier

            assert classifier_name in implemented_classifier_models

            if (classifier_name == 'knn'):
                print("")
            elif (classifier_name == 'svm'):
                print("")
            elif (classifier_name == 'linear'):
                classifier = Linear_classifier(self.train,
                                               self.train_clusters,
                                               n_epochs=3000,
                                               lr=0.001)
            elif (classifier_name == 'fc3'):
                classifier = FC3_classifier(self.train,
                                            self.train_clusters,
                                            n_epochs=3000,
                                            lr=0.001)
            elif (classifier_name == 'cnn'):  # for image data
                batch_size = config.cnn_classifier_batch_size
                is_rgb = config.is_rgb
                classifier = CNN_classifier(self.train,
                                            self.train_clusters,
                                            n_epochs=100,
                                            lr=0.001,
                                            batch_size=batch_size,
                                            is_rgb=is_rgb)

            train_pred = classifier.predict(self.train.cuda(config.device))
            train_accuracy = accuracy_score(train_pred, self.train_clusters)

            print(
                "NN Classifier training accuracy : {}".format(train_accuracy))
            log.info(
                "NN Classifier training accuracy = {}".format(train_accuracy))

            apply_odin(classifier, self.test_in, self.test_out)
            calculate_metric("mnist")


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

    def classify_naive(
            self):  # classify with cluster model using naive ML models

        log = config.logger

        test_predict_list = []

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
        best_auc = 0
        best_combi_f1 = None
        best_combi_auc = None
        best_threshold_f1 = 0
        best_threshold_auc = 0

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
                    auc = roc_auc_score(self.test_y,
                                        (data_predict < j).astype(int))

                    if f1 > best_f1:
                        best_f1 = f1
                        best_combi_f1 = combi_list[combi_]
                        best_threshold_f1 = j

                    if auc > best_auc:
                        best_auc = auc
                        best_combi_auc = combi_list[combi_]
                        best_threshold_auc = j

                    print(
                        "cluster_num - {}, combination - {}, f1 - {:.4f}, threshold - {}"
                        .format(self.cluster_num, combi_list[combi_], f1, j))
                    log.info(
                        "cluster_num - {}, combination - {}, f1 - {:.4f}, threshold - {}"
                        .format(self.cluster_num, combi_list[combi_], f1, j))
                    print(
                        "cluster_num - {}, combination - {}, auc - {:.4f}, threshold - {}"
                        .format(self.cluster_num, combi_list[combi_], auc, j))
                    log.info(
                        "cluster_num - {}, combination - {}, auc - {:.4f}, threshold - {}"
                        .format(self.cluster_num, combi_list[combi_], auc, j))

        print("best f1 score : {:.4f}".format(best_f1))
        log.info("best f1 score : {:.4f}".format(best_f1))
        print("best auc score : {:.4f}".format(best_auc))
        log.info("best auc score : {:.4f}".format(best_auc))
        print("best f1 combi = {}".format(best_combi_f1))
        log.info("best f1 combi = {}".format(best_combi_f1))
        print("best auc combi = {}".format(best_combi_auc))
        log.info("best auc combi = {}".format(best_combi_auc))
        print("best f1 threshold : {}".format(best_threshold_f1))
        log.info("best f1 threshold : {}".format(best_threshold_f1))
        print("best auc threshold : {}".format(best_threshold_auc))
        log.info("best auc threshold : {}".format(best_threshold_auc))
