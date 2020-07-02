"""
configuration setting for etri_2019
"""

import logging
import datetime

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

implemented_datasets = ('swat', 'cola', 'reuters')
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
embedding = 's_bert'
# avg_glove, avg_bert, s_bert,...

# cluster specific configuration
# gaussian mixture model
gmm_type = 'tied'

# -------------------------------
# clustering configuration
# -------------------------------

# deep embedding clustering
dec_pretrain_epochs = 300
dec_finetune_epochs = 500
dec_batch_size = 256

set_dec_lower_learning_rate = False
lower_lr = 0.01
lower_momentum = 0.6

# -------------------------------
# classifier configuration
# -------------------------------

svm_gamma = 0.1
svm_C = 1000

knn_n_neighbors = 10

# lr = 0.001

#  MAX_LEN = 64 # for BERT padding

# CoLA data
# CoLA_num_of_label = 2 # for CoLA dataset label specification
