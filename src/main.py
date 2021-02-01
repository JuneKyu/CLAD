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
from models.my_model import Model

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
    parser.add_argument('--dataset_name', type=str, default='swat')
    parser.add_argument('--sentence_embedding',
                        type=str,
                        default='sentence_embedding')
    parser.add_argument('--normal_class_index_list', nargs='+',
                        default=[0])  # get a list of normal class indexes
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--n_hidden_features', type=int, default=10)
    parser.add_argument('--cluster_type', type=str, default='gmm')
    parser.add_argument('--dec_pretrain_epochs', type=int, default=100)
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
    config.sentence_embeddingm = args.sentence_embedding
    normal_class_index_list = args.normal_class_index_list
    normal_class_index_list = [int(i) for i in normal_class_index_list]
    config.normal_class_index_list = normal_class_index_list
    cluster_num = args.cluster_num
    config.cluster_num = cluster_num
    n_hidden_features = args.n_hidden_features
    config.n_hidden_features = n_hidden_features
    cluster_type = args.cluster_type
    config.cluster_type = cluster_type
    config.dec_pretrain_epochs = args.dec_pretrain_epochs
    config.dec_pretrain_lr = args.dec_pretrain_lr
    config.dec_train_epochs = args.dec_train_epochs
    config.dec_train_lr = args.dec_train_lr
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
    print("classifier : " + classifier)
    log.info("classifier : " + classifier)

    # data specific parameter configurations
    #  if (dataset_name in ("swat")) and (cluster_type in ("dec")):
    #      config.set_dec_lower_learning_rate = True

    #  if (config.dataset_name != '')
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

    model = Model(dataset_name=dataset_name,
                  dataset=dataset,
                  cluster_num=cluster_num,
                  cluster_type=cluster_type,
                  classifier=classifier)

    print("clustering...")
    log.info("clustering...")
    model.cluster()

    print("classifing...")
    log.info("classifing...")
    #  model.classify_naive()
    model.classify_nn(dataset_name)

    log.info("-" * 30)
    log.info("-" * 30)
    log.info('FINISH')
    log.info('%s:%s:%s' %
             (datetime.datetime.now().hour, datetime.datetime.now().minute,
              datetime.datetime.now().second))


if __name__ == "__main__":
    main()
