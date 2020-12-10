from .deep_embedding_clustering import DEC_Module
from .classifiers import KNN_classifier, SVM_classifier, Linear_classifier, FC3_classifier, CNN_classifier, CNN_large_classifier
from config import implemented_cluster_models, implemented_classifier_models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import combinations

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
                 classifier):
        """TODO: to be defined. """

        self.dataset_name = dataset_name
        #  self.dec_train = dataset["dec_train"]
        #  self.dec_train_y = dataset["dec_train_y"]
        self.train_x = dataset["train_x"]
        self.train_y = dataset["train_y"]
        self.test_in = dataset["test_in"]
        self.test_out = dataset["test_out"]

        # cluster variables
        self.cluster_num = cluster_num
        self.cluster_type = cluster_type
        self.train_clusters = []
        self.cluster_model = None

        # classifier variables
        self.classifier_type = classifier

        assert cluster_type in implemented_cluster_models
        assert classifier in implemented_classifier_models

    def cluster(self):
        """
        clustering module of the model
        """

        #  run deep embedding clustering
        #  cluster_model = DEC_Module(
        #      train_x=self.train_x,
        #      train_y=self.train_y,
        #      batch_size=128,
        #      cluster_type=self.cluster_type,
        #      n_components=self.cluster_num,
        #      n_hidden_features=10)  # need to be configurable
        #  # n_hidden_features for mnist is 10
        #  # n_hidden_features for cifar 10 is ? 30
        #
        #  #  cluster_model.pretrain(epochs=500, lr=0.1,
        #  #                         momentum=0.9)  # for linear dec module
        #  cluster_model.pretrain(epochs=300, lr=0.001, momentum=0.9)
        #  cluster_model.train(epochs=100, lr=0.01, momentum=0.9)
        #  self.clusters, _ = cluster_model.predict()
        #  self.clusters = self.clusters.numpy()

        if (os.path.exists(config.temp_dec_cluster)):
            self.clusters = np.load(
                os.path.join(config.temp_dec_cluster, "train_clusters.npy"))
        else:
            cluster_model = DEC_Module(
                dataset_name=self.dataset_name,
                train_x=self.train_x,
                train_y=self.train_y,
                batch_size=config.dec_batch_size,  # 128
                cluster_type=self.cluster_type,
                n_components=self.cluster_num,
                n_hidden_features=config.n_hidden_features)

            #  TODO : hyperparameters should be configured.

            # mnist - dec
            #  cluster_model.pretrain(epochs=100, lr=0.01, momentum=0.9)
            #  cluster_model.train(epochs=100, lr=0.01, momentum=0.9)

            # mnist - conv-dec
            #  print("conv-dec")
            cluster_model.pretrain(epochs=100)
            cluster_model.train(epochs=100)
            self.clusters, _ = cluster_model.predict()
            self.clusters = self.clusters.numpy()
            os.makedirs(config.temp_dec_cluster)
            np.save(os.path.join(config.temp_dec_cluster, "train_clusters"),
                    self.clusters)

    def classify_nn(self, dataset_name):
        log = config.logger
        classifier_name = config.classifier

        assert classifier_name in implemented_classifier_models

        if (dataset_name in config.cps_datasets):
            print("classifier")
            print("epoch: {}".format(config.classifier_epochs))
            print("lr: {}".format(config.classifier_lr))
            if (classifier_name == 'linear'):
                print('linear')
                classifier = Linear_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.classifier_epochs,
                    lr=config.classifier_lr)
            elif (classifier_name == 'fc3'):
                print('fc3')
                classifier = FC3_classifier(self.train_x,
                                            self.clusters,
                                            n_epochs=config.classifier_epochs,
                                            lr=config.classifier_lr)

            #  if (classifier_name == 'linear'):
            #      classifier = Linear_classifier(
            #          self.train_x,
            #          self.clusters,
            #          n_epochs=config.linear_classifier_epochs,
            #          lr=config.linear_classifier_lr)
            #  elif (classifier_name == 'fc3'):
            #      classifier = FC3_classifier(
            #          self.train_x,
            #          self.clusters,
            #          n_epochs=config.fc3_classifier_epochs,
            #          lr=config.fc3_classifier_lr)

        elif (dataset_name in config.text_datasets):
            classifier = Linear_classifier(self.train_x,
                                           self.clusters,
                                           n_epochs=5000,
                                           lr=0.001)
            train_pred = classifier.module.predict(
                self.train_x.cuda(config.device))
            train_accuracy = accuracy_score(train_pred, self.clusters)

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
        elif (dataset_name in config.image_datasets):

            if (classifier_name == 'knn'):
                print("")
            elif (classifier_name == 'svm'):
                print("")
            elif (classifier_name == 'linear'):
                classifier = Linear_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.linear_classifier_epochs,
                    lr=config.linear_classifier_lr)
            elif (classifier_name == 'fc3'):
                classifier = FC3_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.fc3_classifier_epochs,
                    lr=config.fc3_classifier_lr)
            elif (classifier_name == 'cnn'):  # for image data
                batch_size = config.cnn_classifier_batch_size
                is_rgb = config.is_rgb
                classifier = CNN_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.cnn_classifier_epochs,
                    lr=config.cnn_classifier_lr,
                    batch_size=batch_size,
                    is_rgb=is_rgb)
            elif (classifier_name == 'cnn_large'):
                batch_size = config.cnn_large_classifier_batch_size
                is_rgb = config.is_rgb
                classifier = CNN_large_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.cnn_large_classifier_epochs,
                    lr=config.cnn_large_classifier_lr,
                    batch_size=batch_size,
                    is_rgb=is_rgb)

        print("Calculating NN Classifier training accuracy...")
        train_pred = classifier.module.predict(self.train_x.cuda(
            config.device))
        train_accuracy = accuracy_score(train_pred, self.clusters)
        torch.cuda.empty_cache()

        print("NN Classifier training accuracy : {}".format(train_accuracy))
        log.info("NN Classifier training accuracy = {}".format(train_accuracy))
        apply_odin(classifier, self.test_in, self.test_out)
        print("Calculating Metrics")
        calculate_metric("mnist")
