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


def main():

    np.random.seed(777)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str, default='swat')
    parser.add_argument('--normal_class_index', type=int, default='0')
    parser.add_argument('--cluster_num', type=int, default=5)
    parser.add_argument('--cluster_type', type=str, default='gmm')
    parser.add_argument('--classifier_type', type=str, default='svm')
    parser.add_argument('--use_noise_labeling', type=bool, default='True')
    # dataset_name : 'swat', 'wadi', 'cola', 'reuters', 'newsgroups', 'imdb'

    args = parser.parse_args()

    cluster_num = args.cluster_num
    cluster_type = args.cluster_type
    classifier_type = args.classifier_type
    normal_class_index = args.normal_class_index

    # data_path
    data_path = args.data_path
    # data_name
    dataset_name = args.dataset_name

    # logger
    log = config.logger
    folder_path = config.folder_path
    if os.path.exists(folder_path) == False:
        os.makedirs(folder_path)
    fileHandler = logging.FileHandler(\
            os.path.join(folder_path, config.current_time + '-' +\
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
    if dataset_name in ("reuters"):
        print("normal_class_index : {}".format(normal_class_index))
        log.info("normal_class_index : {}".format(normal_class_index))

    if (dataset_name in ("swat")) and (cluster_type in ("dec")):
        config.set_dec_lower_learning_rate = True

    # loading dataset
    dataset = load_dataset(dataset_name=dataset_name,
                           data_path=data_path,
                           normal_class_index=normal_class_index)

    print("")
    print("dataset loading successful!")
    log.info("dataset loading successful")

    model = Model(dataset=dataset,
                  cluster_num=cluster_num,
                  cluster_type=cluster_type,
                  classifier_type=classifier_type,
                  use_noise_labeling=args.use_noise_labeling)

    print("clustering...")
    log.info("clustering...")
    model.cluster()

    print("classifing...")
    log.info("classifing...")
    model.classify()

    log.info("-" * 30)
    log.info("-" * 30)
    log.info('FINISH')
    log.info('%s:%s:%s' %
             (datetime.datetime.now().hour, datetime.datetime.now().minute,
              datetime.datetime.now().second))


if __name__ == "__main__":
    main()
