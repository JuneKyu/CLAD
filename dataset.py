import os 
import pandas as pd
import wget
import zipfile

from transformers import BertTokenizer
from preprocessing import *

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from scipy import signal

import pdb
import argparse

def tokenize_with_bert(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    #  input_ids = []
    #  for sent in sentences:
    #      encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
    #      input_ids.append(encoded_sent)

    tokenized_sentences = np.array( [tokenizer.tokenize(sentence) for sentence in sentences] )

    return tokenized_sentences

def cola_dataset(**kwargs):

    data_path = kwargs['data_path']
    cola_data_path = os.path.join(data_path, 'cola_public')

    if not os.path.exists(cola_data_path):
        print("Downloading CoLA dataset ...")
        url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
        wget.download(url, data_path)
        cola_zip = zipfile.ZipFile(os.path.join(data_path, 'cola_public_1.1.zip'))
        cola_zip.extractall(data_path)

    print("Tokenizing CoLA train data with bert tokenizer")

    cola_train_data_path = os.path.join(cola_data_path, "raw/in_domain_train.tsv")
    train = pd.read_csv(cola_train_data_path, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

    train_x = tokenize_with_bert(train.sentence.values)
    train_y = train.label.values

    print("Tokenizing CoLA val data with bert tokenizer")

    cola_val_data_path = os.path.join(cola_data_path, "raw/in_domain_dev.tsv")
    val = pd.read_csv(cola_val_data_path, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    val_x = tokenize_with_bert(val.sentence.values)
    val_y = val.label.values

    print("Tokenizing CoLA test data with bert tokenizer")

    cola_test_data_path = os.path.join(cola_data_path, "raw/out_of_domain_dev.tsv")
    test = pd.read_csv(cola_test_data_path, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    test_x = tokenize_with_bert(test.sentence.values)
    test_y = test.label.values
   
    return train_x, train_y.astype(int), val_x, val_y.astype(int), test_x, test_y.astype(int)
    


def Wadi_dataset(**kwargs):
    
    data_path = kwargs['data_path']
    train_path = os.path.join(data_path, 'train')
    var_path = os.path.join(data_path, 'test')    
    test_path = os.path.join(data_path, 'test')    

    train_x = pd.read_csv(train_path + '_x.csv')
    train_y = pd.read_csv(train_path  + '_y.csv')
    val_x = pd.read_csv(var_path + '_x.csv')
    val_y = pd.read_csv(var_path  + '_y.csv')
    test_x = pd.read_csv(test_path + '_x.csv')
    test_y = pd.read_csv(test_path  + '_y.csv')    
    
    feature_name = train_x.columns
    
    if 'freq_select_list' in kwargs.keys():
        
        freq_select_list = kwargs['freq_select_list']
    
        freq_train_x = get_freq_data_2(data = train_x, freq_select_list = freq_select_list, read_size = kwargs['read_size'],
                                      window_size = 30)
        freq_val_x = get_freq_data_2(data = val_x, freq_select_list = freq_select_list, read_size = kwargs['read_size'],
                                      window_size = 30)
        freq_test_x = get_freq_data_2(data = test_x, freq_select_list = freq_select_list, read_size = kwargs['read_size'],
                                      window_size = 30)

        train_x = np.concatenate((train_x, freq_train_x),1)
        val_x = np.concatenate((val_x, freq_val_x),1)
        test_x = np.concatenate((test_x, freq_test_x),1)

  
    

    return train_x, train_y.values, val_x, \
            val_y.values, test_x, test_y.values,\
            feature_name



def SWaT_dataset(**kwargs):
    
    data_path = kwargs['data_path']
    train_path = data_path + 'SWaT_Dataset_Normal_v0.csv'
    val_path = data_path + 'SWaT_Dataset_Attack_v0.csv'
    test_path = data_path + 'SWaT_Dataset_Attack_v0.csv'

    
    train = pd.read_csv(train_path, index_col = 0)
    val = pd.read_csv(val_path, index_col = 0)
    test = pd.read_csv(test_path, index_col = 0)
    feature_name = train.columns
    
    if 'freq_select_list' in kwargs.keys():
        freq_select_list = kwargs['freq_select_list']
   
        freq_train_x = get_freq_data_2(data = train, 
                freq_select_list = freq_select_list, read_size = kwargs['read_size'],
                                      window_size = 30)
        freq_val_x = get_freq_data_2(data = val, 
                freq_select_list = freq_select_list, read_size = kwargs['read_size'],
                                      window_size = 30)
        freq_test_x = get_freq_data_2(data = test, 
                freq_select_list = freq_select_list, read_size = kwargs['read_size'],
                                      window_size = 30)

    
    x_col = list(train.columns)[:-1]
    y_col = list(train.columns)[-1]
    train_x = train[x_col]
    val_x = val[x_col]
    test_x = test[x_col]
    
    if 'freq_select_list' in kwargs.keys():
        train_x = np.concatenate((train_x, freq_train_x),1)
        val_x = np.concatenate((val_x, freq_val_x),1)
        test_x = np.concatenate((test_x, freq_test_x),1)
        
    train_y = train[y_col]
    val_y = val[y_col]
    test_y = test[y_col]
    train_y[train_y=='Normal'] = 0
    train_y[train_y=='Attack'] = 1
    val_y[val_y=='Normal'] = 0
    val_y[val_y=='Attack'] = 1
    test_y[test_y=='Normal'] = 0
    test_y[test_y=='Attack'] = 1
    
    
    return train_x, train_y.values.astype(int), val_x, \
            val_y.values.astype(int), test_x, test_y.values.astype(int),\
            feature_name
    
    
    
    
# def get_freq_data(**kwargs):
    
   
#     data = kwargs['data'][kwargs['freq_select_list']]

#     read_size = kwargs['read_size']
    
#     f, t, Sxx = signal.spectrogram(data.values[:,0], 1)

#     tp = np.zeros((data.shape[0]) ,dtype=float)
#     tp[0 : int(t[0])] = np.sum(Sxx[:,0])
#     for i in range(0,Sxx.shape[1] - 1):
#         tp[int(t[i]) : int(t[i+1])] = np.sum(Sxx[0:read_size,i])

#     freq_data = np.copy(tp)
#     for feature in range(1,data.shape[1]):
#         f, t, Sxx = signal.spectrogram(data.values[:,feature], 1)
#         tp = np.zeros((data.shape[0]) ,dtype=float)
#         tp[0 : int(t[0])] = np.sum(Sxx[0:read_size,0])
#         for i in range(0, Sxx.shape[1] - 1):
#             tp[int(t[i]) : int(t[i+1])] = np.sum(Sxx[0:read_size,i])
#         freq_data = np.concatenate((freq_data.reshape(data.shape[0],-1), tp.reshape(-1,1)),1)

#     return freq_data


# freq modifying .. 
def get_freq_data_2(**kwargs):
   
    data = kwargs['data'][kwargs['freq_select_list']]

    read_size = kwargs['read_size']
    
    window_size = kwargs['window_size']
    
    f, t, Sxx = signal.spectrogram(data.values[:,0], 1)

    tp = np.zeros((data.shape[0], read_size) ,dtype=float)
    
    tp[0 : int(t[0]), : ] = Sxx[0:read_size, 0]

    for i in range(0,Sxx.shape[1] - 1):
        tp[int(t[i]) : int(t[i+1])] = np.sum(Sxx[0:read_size,i])

    freq_data = np.copy(tp)
    for feature in range(1,data.shape[1]):
        f, t, Sxx = signal.spectrogram(data.values[:,feature], 1, window = signal.tukey(window_size))
        tp = np.zeros((data.shape[0], read_size) ,dtype=float)
        tp[0 : int(t[0]), : ] = Sxx[0:read_size, 0]
        for i in range(0, Sxx.shape[1] - 1):
            tp[int(t[i]) : int(t[i+1]), : ] = Sxx[0:read_size, i]
        freq_data = np.concatenate((freq_data.reshape(data.shape[0],-1), tp.reshape(data.shape[0],-1)), axis=1)

    return freq_data






# def SWaT_dataset(**kwargs):
    
#     data_path = kwargs['data_path']
#     train_path = data_path + 'SWaT_Dataset_Normal_v0.csv'
#     val_path = data_path + 'SWaT_Dataset_Attack_v0.csv'
#     test_path = data_path + 'SWaT_Dataset_Attack_v0.csv'

    
#     train = pd.read_csv(train_path, index_col = 0)
#     val = pd.read_csv(val_path, index_col = 0)
#     test = pd.read_csv(test_path, index_col = 0)
#     feature_name = train.columns
    
#     if 'freq_select_list' in kwargs.keys():
#         freq_select_list = kwargs['freq_select_list']
   
#         freq_train_x = get_freq_data(data = train, 
#                 freq_select_list = freq_select_list, read_size = kwargs['read_size'])
#         freq_val_x = get_freq_data(data = val, 
#                 freq_select_list = freq_select_list, read_size = kwargs['read_size'])
#         freq_test_x = get_freq_data(data = test, 
#                 freq_select_list = freq_select_list, read_size = kwargs['read_size'])

    
#     x_col = list(train.columns)[:-1]
#     y_col = list(train.columns)[-1]
#     train_x = train[x_col]
#     val_x = val[x_col]
#     test_x = test[x_col]
    
#     if 'freq_select_list' in kwargs.keys():
#         train_x = np.concatenate((train_x, freq_train_x),1)
#         val_x = np.concatenate((val_x, freq_val_x),1)
#         test_x = np.concatenate((test_x, freq_test_x),1)
        
#     train_y = train[y_col]
#     val_y = val[y_col]
#     test_y = test[y_col]
#     train_y[train_y=='Normal'] = 0
#     train_y[train_y=='Attack'] = 1
#     val_y[val_y=='Normal'] = 0
#     val_y[val_y=='Attack'] = 1
#     test_y[test_y=='Normal'] = 0
#     test_y[test_y=='Attack'] = 1
    
    
#     return train_x, train_y.values.astype(int), val_x, \
#             val_y.values.astype(int), test_x, test_y.values.astype(int),\
#             feature_name




























# if __name__ == "__main__":
   
#     np.random.seed(777)

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--selected_dim', nargs='+', type=int, default=[[0,1], [2,3], [0,1,2]])
#     parser.add_argument('--freq_select_list', nargs='+', type=str, default=[['2_FIT_002_PV','P1_LIT101']])

#     args.parser.parse_args()

#     _, _, _, _, _, _ = SWaT_dataset(data_path = './swat_data', freq_select_list = ['P1_LIT101', 'P1_MV101', 'P1_P101', 'P2_P203', 'P3_MV301', 'P4_LIT401', 'P5_AIT503']




    






