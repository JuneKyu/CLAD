#!/home/junekyu/anaconda3/bin python3
# -*- coding: utf-8 -*-
"""
python main.py for anomally detection
paper name : ''
"""
import time
import datetime
import os
import pdb
import argparse
import logging
import numpy as np
import warnings
import config
from data_util.main import load_dataset
from models.clad import CLAD

warnings.filterwarnings('ignore')


# for parsing boolean type parameter
def str2bool(v):
    if (v.lower() in ('true')):
        return True
    elif (v.lower() in ('false')):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    np.random.seed(777)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--normal_class_index_list', nargs='+',
                        default=[0])  # get a list of normal class indexes
    parser.add_argument('--cluster_type', type=str, default='cvae')
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--n_hidden_features', type=int, default=10)
    parser.add_argument('--cluster_model_pretrain_epochs',
                        type=int,
                        default=100)
    parser.add_argument('--cluster_model_pretrain_lr',
                        type=float,
                        default=0.01)
    parser.add_argument('--cluster_model_train_epochs', type=int, default=100)
    parser.add_argument('--cluster_model_train_lr', type=float, default=0.01)
    parser.add_argument('--save_cluster_model', type=str2bool, default=False)
    parser.add_argument('--load_cluster_model', type=str2bool, default=False)
    parser.add_argument('--classifier_type', type=str, default='resnet')
    parser.add_argument('--classifier_epochs', type=int, default=200)
    parser.add_argument('--classifier_lr', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1000)
    parser.add_argument('--perturbation', type=float, default=0.001)
    parser.add_argument('--plot_clustering', type=str2bool, default=False)
    parser.add_argument('--save_classifier_model',
                        type=str2bool,
                        default=False)
    parser.add_argument('--load_classifier_model',
                        type=str2bool,
                        default=False)

    args = parser.parse_args()

    data_path = args.data_path
    dataset_name = args.dataset_name
    # if image data, set rgb flag
    if (dataset_name in config.rgb_datasets):
        config.is_rgb = True
        config.cvae_channel = 3
    normal_class_index_list = args.normal_class_index_list
    normal_class_index_list = [int(i) for i in normal_class_index_list]
    config.normal_class_index_list = normal_class_index_list
    cluster_num = args.cluster_num
    config.cluster_num = cluster_num
    n_hidden_features = args.n_hidden_features
    config.n_hidden_features = n_hidden_features
    cluster_type = args.cluster_type
    config.cluster_type = cluster_type
    config.cluster_model_pretrain_epochs = args.cluster_model_pretrain_epochs
    config.cluster_model_pretrain_lr = args.cluster_model_pretrain_lr
    config.cluster_model_train_epochs = args.cluster_model_train_epochs
    config.cluster_model_train_lr = args.cluster_model_train_lr
    config.save_cluster_model = args.save_cluster_model
    config.load_cluster_model = args.load_cluster_model
    classifier_type = args.classifier_type
    config.classifier_type = classifier_type
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
            classifier_type + '.txt'))
    fileHandler.setFormatter(config.formatter)
    config.logger.addHandler(fileHandler)

    log.info("-" * 99)
    log.info("-" * 10 + str(args) + "-" * 10)
    log.info("-" * 99)
    log.info('START %s:%s:%s\n' %
             (datetime.datetime.now().hour, datetime.datetime.now().minute,
              datetime.datetime.now().second))
    log.info('%s:%s:%s\n' %
             (datetime.datetime.now().hour, datetime.datetime.now().minute,
              datetime.datetime.now().second))

    print("dataset name : " + dataset_name)
    log.info("dataset name : " + dataset_name)
    print("classifier : " + classifier_type)
    log.info("classifier : " + classifier_type)

    print("normal_class_index_list : {}".format(normal_class_index_list))
    log.info("normal_class_index_list : {}".format(normal_class_index_list))
    print("n_hidden_features : {}".format(n_hidden_features))
    log.info("n_hidden_features : {}".format(n_hidden_features))
    print("temperature : {}".format(temperature))
    log.info("temperature : {}".format(temperature))
    print("perturbation : {}".format(perturbation))
    log.info("perturbation : {}".format(perturbation))

    # loading dataset
    dataset = load_dataset(dataset_name=dataset_name, data_path=data_path)

    print("")
    print("dataset loading successful!")
    log.info("dataset loading successful")

    model = CLAD(dataset_name=dataset_name,
                 dataset=dataset,
                 cluster_num=cluster_num,
                 cluster_type=cluster_type,
                 classifier_type=classifier_type)

    print("clustering...")
    log.info("clustering...")
    model.cluster()

    print("classifying...")
    log.info("classifying...")
    model.classify_nn(dataset_name)

    log.info("-" * 30)
    log.info("-" * 30)
    log.info('FINISH')
    log.info('%s:%s:%s' %
             (datetime.datetime.now().hour, datetime.datetime.now().minute,
              datetime.datetime.now().second))


if __name__ == "__main__":
    main()
