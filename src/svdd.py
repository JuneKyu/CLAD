#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import datetime
import os
import pdb
import argparse
import logging
import numpy as np
import warnings
import config
import torch
#  from sklearn.svm import OneClassSVM
from svdd_src.svdd import SVDD
from svdd_src.visualize import Visualization as draw
from data_util.main import load_dataset
from sklearn.metrics import roc_curve, auc

# for parsing boolean type parameter
def str2bool(v):
    if (v.lower() in ('true')):
        return True
    elif (v.lower() in ('false')):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    #  np.random.seed(777)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str, default='swat')
    parser.add_argument('--normal_class_index_list', nargs='+',
                        default=[0])  # get a list of normal class indexes
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--n_hidden_features', type=int, default=10)
    parser.add_argument('--cluster_type', type=str, default='gmm')
    parser.add_argument('--dec_pretrain_lr', type=float, default=0.01)
    parser.add_argument('--dec_train_epochs', type=int, default=100)
    parser.add_argument('--dec_train_lr', type=float, default=0.01)
    parser.add_argument('--save_cluster_model', type=str2bool, default=False)
    parser.add_argument('--load_cluster_model', type=str2bool, default=False)
    parser.add_argument('--classifier', type=str, default='linear')
    parser.add_argument('--classifier_epochs', type=int, default=200)
    parser.add_argument('--classifier_lr', type=float, default=0.01)
    parser.add_argument('--save_classifier_model',
                        type=str2bool,
                        default=False)
    parser.add_argument('--load_classifier_model',
                        type=str2bool,
                        default=False)
    parser.add_argument('--temperature', type=float, default=1000)
    parser.add_argument('--perturbation', type=float, default=0.001)
    parser.add_argument('--plot_clustering', type=str2bool, default=False)
    parser.add_argument('--width_factor', type=float, default=80)

    #  parser.add_argument('--use_noise_labeling', type=bool, default='True')
    # dataset_name : 'swat', 'wadi', 'cola', 'reuters', 'newsgroups', 'imdb'

    args = parser.parse_args()

    data_path = args.data_path
    dataset_name = args.dataset_name
    # if image data, set rgb flag
    if (dataset_name in config.rgb_datasets):
        config.is_rgb = True
        config.cvae_channel = 3
    # if text data, set sentence embedding
    normal_class_index_list = args.normal_class_index_list
    normal_class_index_list = [int(i) for i in normal_class_index_list]
    config.normal_class_index_list = normal_class_index_list
    cluster_num = args.cluster_num
    config.cluster_num = cluster_num
    n_hidden_features = args.n_hidden_features
    config.n_hidden_features = n_hidden_features
    cluster_type = args.cluster_type
    config.cluster_type = cluster_type
    config.save_cluster_model = args.save_cluster_model
    config.load_cluster_model = args.load_cluster_model
    classifier = args.classifier
    config.classifier = classifier
    config.classifier_epochs = args.classifier_epochs
    config.classifier_lr = args.classifier_lr
    config.save_classifier_model = args.save_classifier_model
    config.load_classifier_model = args.load_classifier_model
    temperature = args.temperature
    config.temperature = temperature
    perturbation = args.perturbation
    config.perturbation = perturbation
    config.plot_clustering = args.plot_clustering

    # logger
    log = config.logger
    log_path = config.log_path
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    sub_log_path = config.sub_log_path
    if os.path.exists(sub_log_path) == False:
        os.makedirs(sub_log_path)
    fileHandler = logging.FileHandler(\
            os.path.join(sub_log_path, config.current_time + '-' +\
            dataset_name + '-' +\
            cluster_type + '-' +\
            classifier + '.txt'))
    fileHandler.setFormatter(config.formatter)
    config.logger.addHandler(fileHandler)

    #  log.info("-" * 99)
    #  log.info("-" * 10 + str(args) + "-" * 10)
    #  log.info("-" * 99)
    #  log.info('START %s:%s:%s\n' %
    #           (datetime.datetime.now().hour, datetime.datetime.now().minute,
    #            datetime.datetime.now().second))
    #  log.info('%s:%s:%s\n' %
    #           (datetime.datetime.now().hour, datetime.datetime.now().minute,
    #            datetime.datetime.now().second))

    print("dataset name : " + dataset_name)
    log.info("dataset name : " + dataset_name)
    #  print("classifier : " + classifier)
    #  log.info("classifier : " + classifier)
    #
    # data specific parameter configurations
    #  if (dataset_name in ("swat")) and (cluster_type in ("dec")):
    #      config.set_dec_lower_learning_rate = True

    #  if (config.dataset_name != '')
    print("normal_class_index_list : {}".format(normal_class_index_list))
    log.info("normal_class_index_list : {}".format(normal_class_index_list))
    #  print("n_hidden_features : {}".format(n_hidden_features))
    #  log.info("n_hidden_features : {}".format(n_hidden_features))
    #  print("temperature : {}".format(temperature))
    #  log.info("temperature : {}".format(temperature))
    #  print("perturbation : {}".format(perturbation))
    #  log.info("perturbation : {}".format(perturbation))

    # loading dataset
    dataset = load_dataset(dataset_name=dataset_name, data_path=data_path)

    print("")
    print("dataset loading successful!")
    log.info("dataset loading successful")

    train_x = dataset["train_x"]
    train_y = dataset["train_y"]
    test_in = dataset["test_in"]
    test_out = dataset["test_out"]

    print(dataset_name)
    print(normal_class_index_list)

    parameters = {"positive penalty": 0.9,
                  "negative penalty": [],
                  #  "kernel": {"type": 'gauss', "width": 1/80},
                  "kernel": {"type": 'gauss', "width": 1/args.width_factor},
                  "option": {"display": 'on'}}

    svdd = SVDD(parameters)
    #  cls = OneClassSVM(gamma='auto')
    # train
    #  cls.fit(train_x)

    train_x_list = []
    for x in train_x:
        x = x.view(-1).numpy()
        train_x_list.append(x)

    train_y_list = [0] * len(train_x_list)

    print("fitting to svdd")
    train_x = np.array(train_x_list)
    train_y = np.array(train_y_list).reshape(-1, 1)
    svdd.train(train_x, train_y)

    test_in_pred = []
    for t_i in test_in:
        t_i = t_i.view(-1).numpy()
        test_in_pred.append(t_i)

    #  print("predicting test_in")
    #  test_in_pred = cls.predict(test_in_pred)
    #  test_in_pred = cls.score_samples(test_in_pred)

    test_out_pred = []
    for t_o in test_out:
        t_o = t_o.view(-1).numpy()
        test_out_pred.append(t_o)

    #  print("predicting test_out")
    #  test_out_pred = cls.predict(test_out_pred)
    #  test_out_pred = cls.score_samples(test_out_pred)
    #  testData = test_in_pred.tolist()+test_out_pred.tolist()
    testData = test_in_pred + test_out_pred
    testLabel = [0 for i in range(len(test_in_pred))] + [1 for i in range(len(test_out_pred))]

    testData = np.array(testData)
    testLabel = np.array(testLabel)
    distance, accuracy = svdd.test(testData, testLabel)
    print(accuracy)

    #  test_in_pred = cls.predict(test_in)
    #  test_out_pred = cls.predict(test_out)


    #  cls = OneClassSVM(gamma='auto').fit(X)

    #  model = Model(dataset_name=dataset_name,
    #                dataset=dataset,
    #                cluster_num=cluster_num,
    #                cluster_type=cluster_type,
    #                classifier=classifier)

    #  print("clustering...")
    #  log.info("clustering...")
    #  model.cluster()
    #
    #  print("classifing...")
    #  log.info("classifing...")
    #  #  model.classify_naive()
    #  model.classify_nn(dataset_name)
    #
    #  log.info("-" * 30)
    #  log.info("-" * 30)
    #  log.info('FINISH')
    #  log.info('%s:%s:%s' %
    #           (datetime.datetime.now().hour, datetime.datetime.now().minute,
    #            datetime.datetime.now().second))
    #

if __name__ == "__main__":
    main()

