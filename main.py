
import warnings
warnings.filterwarnings('ignore')

from dataset import *

from cluster import *
from classifier import *
from preprocessing import *
import numpy as np

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import logging
import time
import datetime
from sklearn import svm
from matplotlib import pyplot as plt
import argparse

from scipy import signal
from tapr import *
from util import *

from itertools import combinations
import pdb


def main():
    
    np.random.seed(777)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cov_type', type = str, default = 'full')
    parser.add_argument('--gamma', type = float, default = 0.1)
    parser.add_argument('--C', type = float, default = 1000)
    parser.add_argument('--exp', type = str, default = 'tp')
    parser.add_argument('--selected_dim', nargs='+', type=int, default=[[0,1],[2,3],[3,1,2]])
    parser.add_argument('--swat_freq_select_list', nargs='+', type=str, default=[['2_FIT_002_PV']])
    parser.add_argument('--read_size', type = int, default = 5)

    args = parser.parse_args() 
    
#     주파수 분석을 넣은 변수들
    args.swat_freq_select_list = ['P1_LIT101', 'P1_MV101', 'P1_P101' ,'P2_P203', 'P3_DPIT301', 'P3_LIT301','P3_MV301','P4_LIT401', 'P5_AIT503']
    
#     tapr을 위한 변수들
    delta = 600
    theta = 0.001
    alpha = 1.0
    label = [0, 1]

    ev = TaPR(label, theta, delta)
    ev.load_anomalies( './swat_label.csv')
    
#     로거 
    log = logging.getLogger('log')
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    now = datetime.datetime.now()

    today = '%s-%s-%s'%(now.year, now.month, now.day)
    second = ' %sh%sm%ss'%(now.hour, now.minute, now.second)

    folder_path = './log/' + today

    if os.path.exists(folder_path) == False:
        os.makedirs(folder_path)

    fileHandler = logging.FileHandler(os.path.join(folder_path, args.exp  + '.txt'))

    fileHandler.setFormatter(formatter)
    log.addHandler(fileHandler)
    log.info("-"*99)    
    log.info("-"*10 + str(args) + "-"*10)
    log.info("-"*99) 
    

    swat_dic = dict()
    swat_dic['data_path'] = './swat_data/'

    log.info('START %s:%s:%s\n'%(datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
    

#     pca를 위한 선택 변수들
    raw_selected_dim = [0,1,2,38,39,40]
    freq_selected_dim = [0,1,31,32]
    
    swat_n_cluster_list = list(np.arange(5,12))
    
    
    window_size = 30

    log.info('%s:%s:%s\n'%(datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))


#     데이터셋 불러오는 함수
    train_x_swat, train_y_swat, val_x_swat, val_y_swat, test_x_swat, test_y_swat, _ = SWaT_dataset(data_path = swat_dic['data_path'], read_size = args.read_size)

    train_x_swat_freq = get_freq_data_2(data = train_x_swat, 
                    freq_select_list = args.swat_freq_select_list, read_size = args.read_size,
                                          window_size = window_size)
    val_x_swat_freq = get_freq_data_2(data = val_x_swat, 
                    freq_select_list = args.swat_freq_select_list, read_size = args.read_size,
                                          window_size = window_size)
    test_x_swat_freq = get_freq_data_2(data = test_x_swat, 
                    freq_select_list = args.swat_freq_select_list, read_size = args.read_size,
                                          window_size = window_size)



#     swat 불러온후 pca로 전처리 
    train_x_modify_swat, val_x_modify_swat, test_x_modify_swat = PCA_preprocessing_modify(scaler = 'standard', \
                                                                    train_x = train_x_swat, val_x = val_x_swat, \
                                                                  test_x= test_x_swat, n_neighbors = 4, 
                                                                  n_components = 2, feature_num =2, \
                                                                  selected_dim= raw_selected_dim)

    train_x_modify_swat_freq, val_x_modify_swat_freq, test_x_modify_swat_freq = PCA_preprocessing_modify(scaler = 'standard', \
                                                                    train_x = train_x_swat_freq, val_x = val_x_swat_freq, \
                                                                  test_x= test_x_swat_freq, n_neighbors = 4, 
                                                                  n_components = 2, feature_num =2, \
                                                                  selected_dim= freq_selected_dim)

    optimal_f1_swat_list = []
    optimal_tar_swat_list = []
    optimal_tap_swat_list = []
    optimal_cluster_combi_swat_list = []

# 주파수와 일반 데이터 concate 함.
    train_x_modify_swat = np.concatenate((train_x_modify_swat, train_x_modify_swat_freq),1)
    val_x_modify_swat = np.concatenate((val_x_modify_swat, val_x_modify_swat_freq),1)
    test_x_modify_swat = np.concatenate((test_x_modify_swat, test_x_modify_swat_freq),1)

#     클러스터 개수 설정 
    swat_n_cluster_list = list(np.arange(4,6))

    for swat_n_clusters in swat_n_cluster_list:
        log.info("n_clutser {}".format(swat_n_clusters))

        train_predict_swat_list = []
        test_predict_swat_list = []
        label_for_train_swat_list = []
        label_for_var_swat_list = []
        label_for_test_swat_list = []
        classifier_swat_list = []

        tp_optimal_f1_swat = 0
        tp_optimal_tar_swat = 0
        tp_optimal_tap_swat = 0
        tp_optimal_cluster_combi_swat = 0


#         GMM 클러스터링
        train_label_swat, val_label_swat, test_label_swat, cluster_model_gmm_swat = GaussianMixture_clustering(
            n_clusters=swat_n_clusters,train_x = train_x_modify_swat,val_x= val_x_modify_swat,
            test_x = test_x_modify_swat,covariance_type = 'tied') 

    
        for cluster_num in range(np.unique(train_label_swat)[-1] + 1):
#             각각의 클러스터에 대하여 label을 주는 과정
            label_for_train_swat = (train_label_swat == cluster_num).astype(int)
            label_for_var_swat = (val_label_swat == cluster_num).astype(int)
            label_for_test_swat = (test_label_swat == cluster_num).astype(int)
            
#             가끔씩 클러스터가 모든 데이터에 대해 0을 줄때가 있음 이를 위한 보정 
            if np.unique(label_for_train_swat).shape[0] == 1:
                label_for_train_swat[0] = 1
                label_for_test_swat[0] = 1
                label_for_var_swat[0] = 1


            label_for_train_swat_list.append(label_for_train_swat)
            label_for_var_swat_list.append(label_for_var_swat)
            label_for_test_swat_list.append(label_for_test_swat)

#             분류
            classifier_swat = svm.SVC(gamma = args.gamma, C = args.C)
            classifier_swat.fit(train_x_modify_swat, label_for_train_swat)
            classifier_swat_list.append(classifier_swat)

            pred_train_swat = classifier_swat.predict(train_x_modify_swat)
            pred_test_swat = classifier_swat.predict(test_x_modify_swat)

            test_predict_swat_list.append(pred_test_swat)
            train_predict_swat_list.append(pred_test_swat)

        # swat tapr modified version.
        ev.load_anomalies( './swat_label.csv')

        for combi_num in range(2, swat_n_clusters + 1):
            
#             실제 클러스터중 어느 클러스터를 골라야 좋은지 찾는 과정 
            swat_combi_list = list(combinations(test_predict_swat_list, combi_num))
            combi_list = list(combinations(np.arange(swat_n_clusters), combi_num))

            swat_f_tapr_beta = 0
            swat_f1 = 0
            swat_j = 0
            swat_combi_max = 0

            swat_list = 0
            swat_pr = 0
            swat_re = 0

            swat_tar = 0
            swat_tap = 0

            swat_alpha = 0
            undetected = 0

            swat_j = 0

            for combi_ in range(len(swat_combi_list)):
                swat_predict = 0

                for i in range(len(swat_combi_list[combi_])):
                    swat_predict += swat_combi_list[combi_][i]

                for j in range(1, len(swat_combi_list[combi_]) -1):

#                     예측을 TAPR 계산을 위해 저장하고 불러옴
                    pd.DataFrame((swat_predict < j).astype(int)).to_csv('./prediction.csv',index=False, header=None)
                    swat_list_tp = (swat_predict < j).astype(int)

                    ev.load_predictions('./prediction.csv')

                    tapd_value, _ = ev.TaP_d()
                    tard_value, _ = ev.TaR_d()
                    tapp_value = ev.TaP_p()            
                    tarp_value = ev.TaR_p()

                    tar = 0.5*tard_value + 0.5*tarp_value
                    tap = 0.5*tapd_value + 0.5*tapp_value
                    tp_f1 = f1_score(test_y_swat, (swat_predict < j).astype(int))
                    tp_pr = precision_score(test_y_swat, (swat_predict < j).astype(int))
                    tp_re = recall_score(test_y_swat, (swat_predict < j).astype(int))
#                     어느 클러스터를 활용해야 성능이 높은지 일일히 확인 
                    beta = 1
                    tp_f_tapr_beta = (1+(beta**2))*(tap*tar)/((beta**2)*tap + tar+ 0.0001)
                    
                    if  tp_f_tapr_beta > swat_f_tapr_beta:
                        swat_f_tapr_beta = tp_f_tapr_beta
                        swat_tap = tap
                        swat_tar = tar
                        swat_f1 = tp_f1
                        swat_pr = tp_pr
                        swat_re = tp_re

                        swat_combi_max = combi_list[combi_]

                        swat_list = swat_list_tp
                        undetected = find_attack_num(swat_list, test_y_swat)
                        swat_j = j

            log.info("-"*30) 
            log.info('swat pr {:.4f}, re{:.4f}, f1 {:.4f}, combi_list {}, undetectd {} j {}'.\
                  format(swat_pr, swat_re ,swat_f1, swat_combi_max, undetected, swat_j))
            log.info("f tarp {:.4f} tar {:.4f} tap {:.4f}".format(swat_f_tapr_beta, swat_tar, swat_tap))
            log.info(np.unique(swat_list, return_counts=True))
            log.info("-"*30) 
            if tp_optimal_f1_swat < swat_f1:
                tp_optimal_f1_swat = swat_f1
                tp_optimal_tar_swat = swat_tar
                tp_optimal_tap_swat = swat_tap
                tp_optimal_cluster_combi_swat = swat_combi_max
                
        optimal_f1_swat_list.append(tp_optimal_f1_swat)
        optimal_tar_swat_list.append(tp_optimal_tar_swat)
        optimal_tap_swat_list.append(tp_optimal_tap_swat)
        optimal_cluster_combi_swat_list.append(tp_optimal_cluster_combi_swat)      



    log.info(optimal_f1_swat_list)
    log.info(optimal_tar_swat_list)
    log.info(optimal_tap_swat_list)
    log.info(optimal_cluster_combi_swat_list)
    
    
    log.info("-"*30)
    log.info("-"*30)
    log.info('FINISH')   
    log.info('%s:%s:%s'%(datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))




if __name__ == "__main__":
    main()
