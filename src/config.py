"""
configuration setting for etri_2019
"""

import logging
import datetime

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
now = datetime.datetime.now()
today = '%s-%s-%s'%(now.year, now.month, now.day)
current_time = '%s-%s-%s'%(now.hour, now.minute, now.second)
folder_path = '../../log/' + today



implemented_datasets = ('swat', 'wadi', 'cola', 'reuters', 'newsgroups', 'imdb')
implemented_nlp_embeddings = ('avg_bert')


# data specific configuration
embedding = 'avg_bert'

# lr = 0.001

# swat_freq_select_list = ['P1_LIT101', 'P1_MV101', 'P1_P101' ,'P2_P203', 'P3_DPIT301', 'P3_LIT301','P3_MV301','P4_LIT401', 'P5_AIT503']

#  MAX_LEN = 64 # for BERT padding

# CoLA data
# CoLA_num_of_label = 2 # for CoLA dataset label specification

