from .clustering_model import Clustering_Module
from .confidence_trainers import KNN_classifier, SVM_classifier, Linear_classifier, FC3_classifier, CNN_classifier, CNN_large_classifier, ResNet_classifier
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


class CLAD(object):
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
        self.classifier_type = classifier_type

        assert cluster_type in implemented_cluster_models
        assert classifier_type in implemented_classifier_models

    def cluster(self):
        """
        clustering module of the model
        """

        cluster_model = Clustering_Module(
            dataset_name=self.dataset_name,
            train_x=self.train_x,
            train_y=self.train_y,
            batch_size=config.cluster_model_batch_size,
            cluster_type=self.cluster_type,
            n_components=self.cluster_num,
            n_hidden_features=config.n_hidden_features)

        if (config.load_cluster_model):
            #  cluster_model.encoder.load_state_dict(
            #      torch.load(
            #          os.path.join(config.cluster_model_path,
            #                       'cluster_encoder.pth')))
            #  cluster_model.decoder.load_state_dict(
            #      torch.load(
            #          os.path.join(config.cluster_model_path,
            #                       'cluster_decoder.pth')))
            cluster_model.cm.load_state_dict(
                torch.load(
                    os.path.join(config.cluster_model_path,
                                 'cluster_model.pth')))
            cluster_model.cm.to(config.device)
        else:
            cluster_model.pretrain(epochs=config.cluster_model_pretrain_epochs)
            cluster_model.train(epochs=config.cluster_model_train_epochs)

            if (config.save_cluster_model):
                if (os.path.exists(config.cluster_model_path) == False):
                    os.makedirs(config.cluster_model_path)
                #  torch.save(
                #      cluster_model.encoder.state_dict(),
                #      os.path.join(config.cluster_model_path,
                #                   'cluster_encoder.pth'))
                #  torch.save(
                #      cluster_model.decoder.state_dict(),
                #      os.path.join(config.cluster_model_path,
                #                   'cluster_decoder.pth'))
                torch.save(
                    cluster_model.cm.state_dict(),
                    os.path.join(config.cluster_model_path,
                                 'cluster_model.pth'))

        self.clusters, _ = cluster_model.predict()
        self.clusters = self.clusters.numpy()

    def classify_nn(self, dataset_name):

        #  TODO: implement save / load of classifier model
        log = config.logger
        classifier_type = config.classifier_type
        assert classifier_type in implemented_classifier_models

        if (dataset_name in config.cps_datasets):
            print("classifier")
            print("epoch: {}".format(config.classifier_epochs))
            print("lr: {}".format(config.classifier_lr))
            if (classifier_type == 'linear'):
                print('linear')
                classifier = Linear_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.classifier_epochs,
                    lr=config.classifier_lr)
            elif (classifier_type == 'fc3'):
                print('fc3')
                classifier = FC3_classifier(self.train_x,
                                            self.clusters,
                                            n_epochs=config.classifier_epochs,
                                            lr=config.classifier_lr)
            elif (classifier_type == 'gru'):
                print('gru')

                #  classifier =

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

            if (classifier_type == 'knn'):
                print("")
            elif (classifier_type == 'svm'):
                print("")
            elif (classifier_type == 'linear'):
                classifier = Linear_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.classifier_epochs,
                    lr=config.classifier_lr)
            elif (classifier_type == 'fc3'):
                classifier = FC3_classifier(self.train_x,
                                            self.clusters,
                                            n_epochs=config.classifier_epochs,
                                            lr=config.classifier_lr)
            elif (classifier_type == 'cnn'):  # for image data
                batch_size = config.cnn_classifier_batch_size
                is_rgb = config.is_rgb
                classifier = CNN_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.classifier_epochs,
                    lr=config.classifier_lr,
                    #  n_epochs=config.cnn_classifier_epochs,
                    #  lr=config.cnn_classifier_lr,
                    batch_size=batch_size,
                    is_rgb=is_rgb)
            elif (classifier_type == 'cnn_large'):
                batch_size = config.cnn_large_classifier_batch_size
                is_rgb = config.is_rgb
                classifier = CNN_large_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.classifier_epochs,
                    lr=config.classifier_lr,
                    batch_size=batch_size,
                    is_rgb=is_rgb)
            elif (classifier_type == 'resnet'):
                batch_size = config.resnet_classifier_batch_size
                is_rgb = config.is_rgb
                classifier = ResNet_classifier(
                    self.train_x,
                    self.clusters,
                    n_epochs=config.classifier_epochs,
                    lr=config.classifier_lr,
                    batch_size=batch_size,
                    is_rgb=is_rgb)


        # confidence trainer accuracy
        print("Calculating NN Classifier training accuracy...")
        train_pred = classifier.module.predict(self.train_x.cuda(
            config.device))
        train_accuracy = accuracy_score(train_pred, self.clusters)
        torch.cuda.empty_cache()
        print("NN Classifier training accuracy : {}".format(train_accuracy))
        log.info("NN Classifier training accuracy = {}".format(train_accuracy))



        print("Scailing the confidence outputs")
        apply_odin(classifier, self.test_in, self.test_out)

        print("Calculating Metrics")
        calculate_metric()
