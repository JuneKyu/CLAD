"""
configuration setting for etri_2019
"""
import datetime
import logging
import os

import numpy as np
import torch
import random

cwd = os.getcwd()

# -------------------------------
# randomness control
# -------------------------------

random_seed = 777

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# -------------------------------
# cuda configuration
# -------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#  device = torch.device("cpu")

#  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#  os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# -------------------------------
# logger configuration
# -------------------------------

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
now = datetime.datetime.now()
today = '%s-%s-%s' % (now.year, now.month, now.day)
current_time = '%s-%s-%s' % (now.hour, now.minute, now.second)
log_path = os.path.join(cwd, '../../log/' + today)
sub_log_path = os.path.join(log_path, current_time)
# -------------------------------
# implementation configuration
# -------------------------------
cps_datasets = ('swat')
text_datasets = ('cola', 'reuters')
image_datasets = ('mnist', 'cifar10')
rgb_datasets = ('cifar10')

implemented_datasets = ('swat', 'cola', 'reuters', 'mnist', 'cifar10')
# dataset to implement = 'wadi', 'newsgroups', 'imdb'
implemented_nlp_embeddings = ('avg_bert', 's_bert')
# embeddings to implement = 'avg_glove'
implemented_cluster_models = ('dec', 'cvae_base', 'cvae_large', 'cvae_temp')

implemented_classifier_models = ('knn', 'svm', 'linear', 'fc3', 'cnn',
                                 'cnn_large')
# need to implement more...

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
plot_clustering = False
#
cluster_type = 'dec'

# temp dir for debugging
#  temp_dec_cluster = "/home/junekyu/hdd/temp_dec/"
temp_dec_cluster = os.path.join(cwd, "../../data/temp_dec/")
plot_path = os.path.join(sub_log_path, "clustering_plot")

cluster_num = 10
#  dec_batch_size = 128
dec_batch_size = 512
n_hidden_features = 10

# pretrain
dec_pretrain_epochs = 300
dec_pretrain_lr = 0.1
dec_pretrain_momentum = 0.9
#  dec_pretrain_decay_step = 100
# dec_pretrain_decay_rate = 0.1

# finetune

# dec training stage
dec_pretrain_epochs = 100
dec_pretrain_lr = 0.01
dec_train_epochs = 100
dec_train_lr = 0.01
#  dec_train_momentum = 0.9

# reuters
#  reuters_dec_finetune_epochs = 800
#  reuters_dec_finetune_lr = 0.3
#  reuters_dec_finetune_decay_step = 200
#  reuters_dec_finetune_decay_rate = 0.5
#  reuters_dec_train_epochs = 1000

# swat
#  swat_dec_pretrain_epochs = 200
#  swat_dec_lr = 1e-2
#  swat_dec_lr_train = 1e-2
#  swat_dec_momentum = 0.6
#  swat_dec_momentum_train = 0.9

# cifar10
#  cifar10_dec_pretrain_epochs = 200
#  cifar10_dec_finetune_epochs = 1200
#  cifar10_dec_finetune_lr = 0.03  # 0.1 is default
#  cifar10_dec_finetune_momentum = 0.6  # 0.9 is default
#  cifar10_dec_finetune_decay_step = 400
#  cifar10_dec_finetune_decay_rate = 0.5
#  cifar10_dec_train_epochs = 500
#  cifar10_dec_train_lr = 0.001

# cvae + dec_clustering
cvae_channel = 1
cvae_z_dim = 128
cvae_kernel_size = 3
cvae_height = 28
cvae_width = 28

# meanshift clustering

ms_quantile = 0.2
ms_n_samples = 500

# -------------------------------
# classifier configuration
# -------------------------------

classifier = 'linear'
#  implemented_classifiers = ('knn', 'svm', 'linear', 'fc3', 'cnn')

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

classifier_epochs = 200
classifier_lr = 0
""" linear """
# for swat
linear_classifier_epochs = 5000
linear_classifier_lr = 0.0001
# for mnist
#  linear_classifier_epochs = 200
#  linear_classifier_lr = 0.0001
# for cifar10
#  linear_classifier_epochs = 5000
#  linear_classifier_lr = 0.001
""" fc3 """
# mnist
#  fc3_classifier_epochs = 100
#  fc3_classifier_lr = 0.00001
#  for cifar10
fc3_classifier_epochs = 1000
fc3_classifier_lr = 0.001
""" cnn """
# for mnist
#  cnn_classifier_batch_size = 100
#  cnn_classifier_epochs = 100
#  cnn_classifier_lr = 0.00001
#  is_rgb = False
# for cifar10
cnn_classifier_batch_size = 128
cnn_classifier_epochs = 100
cnn_classifier_lr = 0.00001
is_rgb = False

#  cnn_large_classifier_batch_size = 100
#  cnn_large_classifier_epochs = 100
#  cnn_large_classifier_lr = 0.0001
#

# -------------------------------
# ood detector configuration
# -------------------------------

# odin softmax files dir
sf_scores_path = '../softmax_scores'
base_in_path = os.path.join(sf_scores_path, 'confidence_Base_In.txt')
base_out_path = os.path.join(sf_scores_path, 'confidence_Base_Out.txt')
odin_in_path = os.path.join(sf_scores_path, 'confidence_Odin_In.txt')
odin_out_path = os.path.join(sf_scores_path, 'confidence_Odin_Out.txt')

# original temper and perterbation magintude
#  odin_temperature = 1000
#  odin_perturbation_magnitude = 0.0012  # perturbation
temperature = 1000
perturbation = 0.1
# odin with temper 10, perturbation 0.12 : odin 0.9980 , base 0.4313
