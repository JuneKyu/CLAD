"""

Code for ..

"""

import pdb
import sys
import os 
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

        
import sklearn.preprocessing
from sklearn.decomposition import PCA

class triplet_dataset(Dataset):
    def __init__(self, **kwargs):
        self.data_path = kwargs['data_path']
        self.train_path = os.path.join(self.data_path, 'train')
        self.test_path = os.path.join(self.data_path, 'test')
        self.train = kwargs['train']
        
                                       
        self.window_length = kwargs['window_length']       
              
        if kwargs['nrow'] != -1:
            self.nrow = kwargs['nrow']
            self.train_data = pd.read_csv(self.train_path + '_x.csv', nrows = self.nrow)
            self.train_label = pd.read_csv(self.train_path  + '_y.csv', nrows = self.nrow)
            self.test_data = pd.read_csv(self.test_path + '_x.csv', nrows = self.nrow)
            self.test_label = pd.read_csv(self.test_path  + '_y.csv', nrows = self.nrow)            
            
        else: 
            self.train_data = pd.read_csv(self.train_path + '_x.csv')
            self.train_label = pd.read_csv(self.train_path  + '_y.csv')
            self.test_data = pd.read_csv(self.test_path + '_x.csv')
            self.test_label = pd.read_csv(self.test_path  + '_y.csv') 
            if self.train == True:
                self.nrow = self.train_data.shape[0]
            else:
                self.nrow = self.test_data.shape[0]

        
        self.negative_sample_index = np.where(self.test_label == 1)[0]
        
        scaler = sklearn.preprocessing.StandardScaler()

        scaler.fit(self.train_data.values)

        self.train_data = scaler.transform(self.train_data.values)
        self.test_data = scaler.transform(self.test_data.values)
        
        pca = PCA(n_components = np.shape(self.train_data)[1])
        
        pca.fit(self.train_data)
        

        
        self.train_data = pca.transform(self.train_data)
        self.test_data = pca.transform(self.test_data)
        
        self.train_data = torch.from_numpy(self.train_data)
        self.test_data = torch.from_numpy(self.test_data)
        
        self.train_label = torch.from_numpy(self.train_label.values)
        self.test_label = self.test_label.values
        self.train_data = self.train_data[:, 0:8]
        self.test_data = self.test_data[:, 0:8]
        

        
        

    def __getitem__(self, index):
        
        if self.train == True:
            anchor = self.train_data[index * self.window_length \
                                     :index * self.window_length + self.window_length]
            positive_sample =  self.train_data[index * self.window_length + self.window_length :\
                                               index * self.window_length + 2 * self.window_length]
            negative_index = np.random.choice(self.negative_sample_index)
            negative_sample = self.test_data[negative_index :negative_index +\
                                             self.window_length]
            return anchor, positive_sample, negative_sample
        else:
#             print("index {}, unique {}".format(index * self.window_length, np.unique(self.test_label[index * self.window_length : index * self.window_length + self.window_length]).shape[0]))
            if np.unique(self.test_label[index * self.window_length : index * self.window_length\
                                  + self.window_length]).shape[0] == 1 and  np.unique(self.test_label[index * self.window_length : index * self.window_length\
                                  + self.window_length])[0] == 0:
                print("normal  index {}".format(index * self.window_length))
                return self.test_data[index * self.window_length :index * self.window_length\
                                  + self.window_length], 0
            else:
                print("abnormal situation index {}".format(index * self.window_length))
                return self.test_data[index * self.window_length :index * self.window_length\
                                  + self.window_length], 1
            

    def __len__(self):
#         return self.nrow - 2 * self.window_length
        return int(self.nrow / self.window_length) -  2 * self.window_length
    
    

    
    
class triplet_dataset_swat(Dataset):
    def __init__(self, **kwargs):
        self.data_path = kwargs['data_path']
        self.train_path = self.data_path + 'SWaT_Dataset_Normal_v0.csv'
        self.test_path = self.data_path + 'SWaT_Dataset_Attack_v0.csv'
        self.train = kwargs['train']
        
                                       
        self.window_length = kwargs['window_length']       
        train_data = pd.read_csv(self.train_path, index_col = 0)
        test_data = pd.read_csv(self.test_path, index_col = 0)
        x_col = list(train_data.columns)[:-1]
        y_col = list(train_data.columns)[-1]
        self.train_data = train_data[x_col]
        self.test_data = test_data[x_col]

        self.train_label = train_data[y_col]
        self.test_label = test_data[y_col]
        
        self.train_label[self.train_label=='Normal'] = 0
        self.train_label[self.train_label=='Attack'] = 1

        self.test_label[self.test_label=='Normal'] = 0
        self.test_label[self.test_label=='Attack'] = 1
        
       
        self.negative_sample_index = np.where(self.test_label == 1)[0]
#         np.random.choice(self.negative_sample_index)
        
        
        scaler = sklearn.preprocessing.StandardScaler()

        scaler.fit(self.train_data.values)

        self.train_data = scaler.transform(self.train_data.values)
        self.test_data = scaler.transform(self.test_data.values)
        
        pca = PCA(n_components = np.shape(self.train_data)[1])
        
        pca.fit(self.train_data)
        

        
        self.train_data = pca.transform(self.train_data)
        self.test_data = pca.transform(self.test_data)
        
        self.train_data = torch.from_numpy(self.train_data)
        self.test_data = torch.from_numpy(self.test_data)
        
        self.train_label = self.train_label.values
        self.test_label = self.test_label.values
        self.train_data = self.train_data[:, 0:4]
        self.test_data = self.test_data[:, 0:4]
        
        if self.train == True:
            self.nrow = self.train_data.shape[0]
        else:
            self.nrow = self.test_data.shape[0]
        

        
        

    def __getitem__(self, index):
        
        if self.train == True:
            anchor = self.train_data[index * self.window_length \
                                     :index * self.window_length + self.window_length]
            positive_sample =  self.train_data[index * self.window_length + self.window_length :\
                                               index * self.window_length + 2 * self.window_length]
            negative_index = np.random.choice(self.negative_sample_index)
            negative_sample = self.test_data[negative_index :negative_index +\
                                             self.window_length]
            return anchor, positive_sample, negative_sample
        else:
            
            if np.unique(self.test_label[index * self.window_length : index * self.window_length\
                                  + self.window_length]).shape[0] == 1:
                 return self.test_data[index * self.window_length :index * self.window_length\
                                  + self.window_length], 0
            else:
                return self.test_data[index * self.window_length :index * self.window_length\
                                  + self.window_length], 1
            

    def __len__(self):
#         return self.nrow - 2 * self.window_length
        return int(self.nrow / self.window_length) -  2 * self.window_length




class base_dataset(Dataset):
    def __init__(self, **kwargs):
        self.data_path = kwargs['data_path']
        self.window_length = kwargs['window_length']       
       
        if kwargs['train'] == True:
            self.label = pd.read_csv(self.data_path + 'train_y.csv')
            self.file_name = 'train'
        else:
            self.data = pd.read_csv(self.data_path + 'test_y.csv')
            self.file_name = 'test'
        
        if 'nrow' in kwargs.keys():
            self.nrow = kwargs['nrow']
            self.data = pd.read_csv(self.data_path + self.file_name + '_x.csv', nrows = self.nrow)
            self.label = pd.read_csv(self.data_path + self.file_name + '_y.csv', nrows = self.nrow)
        else: 
            self.nrow = len(self.data_y)
            self.data = pd.read_csv(self.data_path + self.file_name + '_x.csv')

        scaler = sklearn.preprocessing.StandardScaler()

        scaler.fit(self.data.values)

        self.data = scaler.transform(self.data.values)
        pca = PCA(n_components = np.shape(self.data)[1])
        pca.fit(self.data)
        self.data = pca.transform(self.data)
        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label.values)
        self.data = self.data[:, 0:2]

    def __getitem__(self, index):
        for i in range(self.window_length):
            if self.label[index + i] != 0:
                return self.data[index:index + self.window_length], self.label[index + i]
        
        return self.data[index:index + self.window_length], self.label[index]

    def __len__(self):
        return self.nrow - self.window_length

def make_dataset(data_path, isprint=False):
    
    train_path = data_path + 'WADI_train.csv'
    test_path = data_path + 'WADI_test.csv'

    train_data = pd.read_csv(train_path, index_col = 0)
    test_data = pd.read_csv(test_path, index_col = 0)

    # Change col name
    col_name = []
    for i in range(len(test_data.columns)):
        col_name.append(test_data.columns[i].split('LOG_DATA\\')[-1])

    train_data.columns = col_name
    test_data.columns = col_name


    # Remove std = 0 
    data_info = train_data.describe()
    
    remove = []
    for i in range(len(train_data.columns) -2): #data and Time  
        if data_info.iloc[2][i] == 0.0:
            remove.append(train_data.columns[i+2])

    for i in range(len(remove)):
        del train_data[remove[i]]
        del test_data[remove[i]]


    # Make Label
    label_info = pd.read_csv(data_path + 'label2.txt', index_col = None,\
            header = None, names=['Date', 'Start_Time', 'End_Time'])

    j = 0

    abnormal_time = []
    test_day = label_info.iloc[j]["Date"].split('-')[0]
    test_start_time = label_info.iloc[j]["Start_Time"].split(':')[0:-1]
    test_end_time = label_info.iloc[j]["End_Time"].split(':')[0:-1]

    for i in range(172801):
        time = test_data.iloc[i]["Time"].replace('.', ':').split(':')[0:-1]
        day = test_data.iloc[i]["Date"].replace('.', ':').split('/')[1]

        if day == test_day and test_start_time == time:
            abnormal_time.append(i)
        elif day == test_day and test_end_time == time:
            abnormal_time.append(i)
            j += 1   
            try:
                test_day = label_info.iloc[j]["Date"].split('-')[0]
                test_start_time = label_info.iloc[j]["Start_Time"].split(':')[0:-1]
                test_end_time = label_info.iloc[j]["End_Time"].split(':')[0:-1]
            except:
                break

    # Remove 겹치는 부분 
    abnormal_time.remove(26099)

    train_label = np.zeros(1048571, dtype=int).reshape(-1,1)
    test_label = np.zeros(172802, dtype=int).reshape(-1,1)

    j = 0
    for i in range(0,len(abnormal_time),2):
        test_label[abnormal_time[i]:abnormal_time[i+1]] = 1

    test_label = pd.Series(test_label.reshape(-1))

    train_data['label'] = train_label
    test_data['label'] = test_label

    x_col = list(train_data.columns)[:-1]
    y_col = list(train_data.columns)[-1]

    train_x = train_data[x_col]
    test_x = test_data[x_col]

    train_y = train_data[y_col]
    test_y = test_data[y_col]

    if isprint == True :
        unique, counts = np.unique(train_y, return_counts=True)
        print('In train :', dict(zip(unique, counts)))
        unique, counts = np.unique(test_y, return_counts=True)
        print('In test :', dict(zip(unique, counts)))     
        print("Feature # = {}".format(len(x_col)))

    del train_x['Date']
    del test_x['Date']
    del train_x['Time']
    del test_x['Time']
    
    pd.DataFrame(train_x.values).to_csv('../wadi_data/train_x.csv', header=train_x.columns, index = None )
    pd.DataFrame(train_y.values).to_csv('../wadi_data/train_y.csv', header=['label'],index = None)
    pd.DataFrame(test_x.values).to_csv('../wadi_data/test_x.csv', header=train_x.columns,index = None )
    pd.DataFrame(test_y.values).to_csv('../wadi_data/test_y.csv', header=['label'],index = None )
    

"""
    # For pytorh                             
    train_x = torch.FloatTensor(train_x.values).cuda()
    train_y = torch.FloatTensor(train_y.values).cuda()
                             
    test_x = torch.FloatTensor(test_x.values).cuda()
    test_y = torch.FloatTensor(test_y.values).cuda()
    pdb.set_trace()

    return train_x, train_y, test_x, test_y
"""

if __name__ == '__main__':
#    make_dataset('../wadi_data/', isprint = True)

    dataset = triplet_dataset_swat(data_path = '../wadi_data/', window_length = 12, nrow = -1, train = True)
    train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = True, num_workers = 2)
    
    # device = torch.device('cuda' if torch.cuda.is_available else "cpu")
    device = 'cuda'
    
    # encoder = model.Encoder("as")
    # encoder.cuda()

    # decoder = model.Decoder("aa")
    # decoder.cuda()



    for epoch in range(2):
        for i, data in enumerate(train_loader):
#             pdb.set_trace()
            anchor, positive, negative = data
            pdb.set_trace()
    pdb.set_trace()

        
        



