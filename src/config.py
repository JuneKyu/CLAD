"""
configuration setting for etri_2019
"""
import datetime
import logging
import os

import torch

# -------------------------------
# cuda configuration
# -------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# -------------------------------
# logger configuration
# -------------------------------

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
now = datetime.datetime.now()
today = '%s-%s-%s' % (now.year, now.month, now.day)
current_time = '%s-%s-%s' % (now.hour, now.minute, now.second)
folder_path = '../../log/' + today

# -------------------------------
# implementation configuration
# -------------------------------
cps_datasets = ('swat')
text_datasets = ('cola', 'reuters')
image_datasets = ('mnist', 'cifar10')

implemented_datasets = ('swat', 'cola', 'reuters', 'mnist', 'cifar10')
# dataset to implement = 'wadi', 'newsgroups', 'imdb'
implemented_nlp_embeddings = ('avg_bert', 's_bert'
                              )  # need to implement more...
# embeddings to implement = 'avg_glove'
implemented_cluster_models = ('gmm', 'dec')  # need to implement more...
# cluster_models to implement = ''
implemented_classifier_models = ('knn', 'svm')  # need to implement more...

# -------------------------------
# data configuration
# -------------------------------

# cpr data specific comfiguration : (swat, wadi)
read_size = 5
window_size = 30
swat_selected_list = [
    'P1_LIT101', 'P1_MV101', 'P1_P101', 'P2_P203', 'P3_DPIT301', 'P3_LIT301',
    'P3_MV301', 'P4_LIT401', 'P5_AIT503'
]
swat_raw_selected_dim = [0, 1, 2, 38, 39, 40]
swat_freq_selected_dim = [0, 1, 31, 32]

# nlp data specific configuration
embedding = 's_bert'  # default
# avg_glove, avg_bert, s_bert,...

normal_class_index_list = [0]
# reuters, mnist congigured
# need to configure the rest

# -------------------------------
# clustering configuration
# -------------------------------

# cluster specific configuration

# gaussian mixture model
gmm_type = 'tied'

#
""" deep embedding clustering """
#

# temp dir for debugging
#  temp_dec_cluster = "/home/junekyu/hdd/temp_dec/"
temp_dec_cluster = "/home/junekyu/Study/nsr/etri_2019/data/temp_dec/"

dec_batch_size = 256

# pretrain
dec_pretrain_epochs = 300
dec_pretrain_lr = 0.1
dec_pretrain_momentum = 0.9
#  dec_pretrain_decay_step = 100
# dec_pretrain_decay_rate = 0.1

# finetune
dec_finetune_epochs = 500
dec_finetune_lr = 0.1
dec_finetune_momentum = 0.9
dec_finetune_decay_step = 100
dec_finetune_decay_rate = 0.1

# dec training stage
dec_train_epochs = 100
dec_train_lr = 0.01
dec_train_momentum = 0.9

# reuters
reuters_dec_finetune_epochs = 800
reuters_dec_finetune_lr = 0.3
reuters_dec_finetune_decay_step = 200
reuters_dec_finetune_decay_rate = 0.5
reuters_dec_train_epochs = 1000

# swat
#  swat_dec_pretrain_epochs = 200
#  swat_dec_lr = 1e-2
#  swat_dec_lr_train = 1e-2
#  swat_dec_momentum = 0.6
#  swat_dec_momentum_train = 0.9

# cifar10
cifar10_dec_pretrain_epochs = 200
cifar10_dec_finetune_epochs = 1200
cifar10_dec_finetune_lr = 0.03  # 0.1 is default
cifar10_dec_finetune_momentum = 0.6  # 0.9 is default
cifar10_dec_finetune_decay_step = 400
cifar10_dec_finetune_decay_rate = 0.5
cifar10_dec_train_epochs = 500
cifar10_dec_train_lr = 0.001

# meanshift clustering

ms_quantile = 0.2
ms_n_samples = 500

# -------------------------------
# classifier configuration
# -------------------------------

svm_gamma = 0.1
svm_C = 1000

knn_n_neighbors = 10

# text classifier (2-layer GRU)
text_classifier_input_size = 256  # need to be fixed
text_classifier_hidden_size = 256
text_classifier_output_size = 256
text_classifier_lr = 0.001
text_classifier_epoch = 10

text_classifier_batch_size = 1024

# -------------------------------
# ood detector configuration
# -------------------------------

# odin softmax files dir
sf_scores_path = '../softmax_scores'
base_in_path = os.path.join(sf_scores_path, 'confidence_Base_In.txt')
base_out_path = os.path.join(sf_scores_path, 'confidence_Base_Out.txt')
odin_in_path = os.path.join(sf_scores_path, 'confidence_Odin_In.txt')
odin_out_path = os.path.join(sf_scores_path, 'confidence_Odin_Out.txt')

# original temper and magintude
#  odin_temperature = 1000
#  odin_perturbation_magnitude = 0.0012  # perturbation
odin_temperature = 1000
odin_perturbation_magnitude = 0.12

# odin with temper 10, perturbation 0.12 : odin 0.9980 , base 0.4313
