#  from .cluster_model import clustering
from .deep_embedding_clustering import DEC_Module
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

        #  #  if self.cluster_type == 'dec' or 'cvae':  # deep embedding clustering
        #  if (self.dataset_name == 'mnist'):
        #      # use dec default configuration
        #      print("")
        #  elif (self.dataset_name == 'cifar10'):
        #      config.dec_pretrain_epochs = config.cifar10_dec_pretrain_epochs
        #      config.dec_finetune_epochs = config.cifar10_dec_finetune_epochs
        #      config.dec_finetune_lr = config.cifar10_dec_finetune_lr
        #      config.dec_finetune_momentum = config.cifar10_dec_finetune_momentum
        #      config.dec_finetune_decay_step = config.cifar10_dec_finetune_decay_step
        #      config.dec_finetune_decay_rate = config.cifar10_dec_finetune_decay_rate
        #      config.dec_train_epochs = config.cifar10_dec_train_epochs
        #      config.dec_train_lr = config.cifar10_dec_train_lr
        #  elif (self.dataset_name == 'reuters'):
        #      config.dec_finetune_epochs = config.reuters_dec_finetune_epochs
        #      config.dec_finetune_lr = config.reuters_dec_finetune_lr
        #      config.dec_finetune_decay_step = config.reuters_dec_finetune_decay_step
        #      config.dec_finetune_decay_rate = config.reuters_dec_finetune_decay_rate
        #      config.dec_train_epochs = config.reuters_dec_train_epochs

        #  run deep embedding clustering
        cluster_model = DEC_Module(
            train_x=self.train_x,
            train_y=self.train_y,
            batch_size=128,
            cluster_type=self.cluster_type,
            n_components=self.cluster_num,
            n_hidden_features=10)  # need to be configurable
        cluster_model.pretrain(epochs=500, lr=0.1, momentum=0.9)
        cluster_model.train(epochs=100, lr=0.01, momentum=0.9)
        self.clusters, _ = cluster_model.predict()
        self.clusters = self.clusters.numpy()

        #  if os.path.exists(config.temp_dec_cluster):
        #      self.clusters = np.load(
        #          os.path.join(config.temp_dec_cluster, "train_clusters.npy"))
        #  else:
        #      os.makedirs(config.temp_dec_cluster)
        #      cluster_model = DEC_Module(
        #          train_x=self.train_x,
        #          train_y=self.train_y,
        #          batch_size=128,
        #          cluster_type=self.cluster_type,
        #          n_components=self.cluster_num,
        #          n_hidden_features=10)  # need to be configurable
        #      cluster_model.pretrain(epochs=500, lr=0.1, momentum=0.9)
        #      cluster_model.train(epochs=100, lr=0.01, momentum=0.9)
        #      self.clusters, _ = cluster_model.predict()
        #      self.clusters = self.cluster.numpy()
        #      np.save(os.path.join(config.temp_dec_cluster, "train_clusters"),
        #              self.clusters)

    def classify_nn(self, dataset_name):
        log = config.logger
        if dataset_name in config.cps_datasets:
            print("not implemented")
        elif dataset_name in config.text_datasets:
            classifier = Linear_classifier(self.train_x,
                                           self.clusters,
                                           n_epochs=5000,
                                           lr=0.001)
            train_pred = classifier.predict(self.train_x.cuda(config.device))
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
        elif dataset_name in config.image_datasets:

            classifier_name = config.classifier

            assert classifier_name in implemented_classifier_models

            if (classifier_name == 'knn'):
                print("")
            elif (classifier_name == 'svm'):
                print("")
            elif (classifier_name == 'linear'):
                classifier = Linear_classifier(self.train_x,
                                               self.clusters,
                                               n_epochs=3000,
                                               lr=0.001)
            elif (classifier_name == 'fc3'):
                classifier = FC3_classifier(self.train_x,
                                            self.clusters,
                                            n_epochs=3000,
                                            lr=0.001)
            elif (classifier_name == 'cnn'):  # for image data
                batch_size = config.cnn_classifier_batch_size
                is_rgb = config.is_rgb
                pdb.set_trace()
                classifier = CNN_classifier(self.train_x,
                                            self.clusters,
                                            n_epochs=100,
                                            lr=0.001,
                                            batch_size=batch_size,
                                            is_rgb=is_rgb)

            train_pred = classifier.predict(self.train_x.cuda(config.device))
            train_accuracy = accuracy_score(train_pred, self.clusters)

            print(
                "NN Classifier training accuracy : {}".format(train_accuracy))
            log.info(
                "NN Classifier training accuracy = {}".format(train_accuracy))
            apply_odin(classifier, self.test_in, self.test_out)
            calculate_metric("mnist")
