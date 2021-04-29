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
torch.cuda.manual_seed(random_seed)
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
image_datasets = ('mnist', 'gtsrb', 'cifar10', 'tiny_imagenet')
rgb_datasets = ('gtsrb', 'cifar10', 'tiny_imagenet')

implemented_datasets = ('mnist', 'gtsrb', 'cifar10', 'tiny_imagenet')
implemented_cluster_models = ('dec', 'cvae', 'cvae_large')

implemented_classifier_models = ('linear', 'fc3', 'cnn',
                                 'cnn_large', 'resnet')

# -------------------------------
# data configuration
# -------------------------------

normal_class_index_list = [0]

# -------------------------------
# clustering configuration
# -------------------------------

save_cluster_model = False
load_cluster_model = False
cluster_model_path = os.path.join(cwd, '../../cluster_model_ckp')

# cluster specific configuration
""" clustering """
plot_clustering = False

#  default clustering
cluster_type = 'cvae'

# temp dir for debugging
plot_path = os.path.join(sub_log_path, "clustering_plot")

cluster_num = 10
cluster_model_batch_size = 512
n_hidden_features = 10

# pretrain
cluster_model_pretrain_epochs = 300
cluster_model_pretrain_lr = 0.1
# finetune
cluster_model_train_epochs = 100
cluster_model_train_lr = 0.01

# cvae + dec_clustering
cvae_channel = 1
cvae_z_dim = 128
cvae_kernel_size = 3
cvae_height = 28
cvae_width = 28

# -------------------------------
# classifier configuration
# -------------------------------

save_classifier_model = False
load_classifier_model = False
classifier_model_path = os.path.join(cwd, '../../classifier_model_ckp')

#  default classifier type
classifier_type = 'resnet'

classifier_epochs = 200
classifier_lr = 0
""" linear """
linear_classifier_epochs = 5000
linear_classifier_lr = 0.0001
""" fc3 """
fc3_classifier_epochs = 1000
fc3_classifier_lr = 0.001
""" cnn """
cnn_classifier_batch_size = 128
cnn_classifier_epochs = 100
cnn_classifier_lr = 0.00001
is_rgb = False

#  cnn_large_classifier_batch_size = 100
#  cnn_large_classifier_epochs = 100
#  cnn_large_classifier_lr = 0.0001

""" resnet """
resnet_classifier_batch_size = 128
resnet_classifier_epochs = 100
resnet_classifier_lr = 0.001

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
