#!/bin/sh


# 0216 swat test 명령어
python main_1018_swat_f1_주파수.py --cov_type 'tied' --gamma 0.1 --C 1000.0 \
--exp 'swat 테스트'


























# python main.py --cov_type 'full' --gamma 0.1 --C 1000.0 --exp 'find clustering'
# python main.py --cov_type 'full' --gamma 0.5 --C 1000.0 --exp 'find clustering'
# python main.py --cov_type 'full' --gamma 0.5 --C 2000.0 --exp 'find clustering'


# python main.py --cov_type 'full' --gamma 0.1 --C 1000.0 \
# --exp 'feqtuure slected modify'\
# --selected_dim \
# --freq_select_list 'P1_LIT101'  'P1_MV101'  'P1_P101'  'P2_P203'  'P3_DPIT301'  'P3_LIT301' 'P3_MV301' 'P4_LIT401'  'P5_AIT503'




# 1011 PCA concat parameter searching 1012 일의 경우 for문 바꿔서 같은 문서로 진행함. 
# python main_pca_concat_1011_swat.py --cov_type 'full' --gamma 0.1 --C 1000.0 \
# --exp 'pca concat 관련 parameter 탐색'




# 1014 PCA concat wadi 다른파라미터는 고정, 클러스터 개수만 찾는 중
# python main_pca_concat_1014_wadi_log.py --cov_type 'tied' --gamma 0.1 --C 1000.0 --exp "wadi Find other parameter"


# 1014 swat tap tar f score 기반으로 찾는 중
# python main_1014_swat_tar_tap_f_score.py --cov_type 'full' --gamma 0.1 --C 1000.0 \
# --exp 'swat tap tar f score 기반으로 찾는 중'


# 1017 wadi 제대로 돌아가나 테스트중 목표는 0.3 이상
# python main_wadi_성능test_freq없이_1017.py --cov_type 'spherical' --gamma 0.1 --C 1000.0 \
# --exp 'wadi 제대로 돌아가나 테스트중 목표는 0.3 이상 freq데이터 없어'



# 1018 swat 최적 freq 찾기
# python main_1018_swat_f1_주파수.py --cov_type 'tied' --gamma 0.1 --C 1000.0 \
# --exp 'wat 최적 freq 찾기'


# 1019 wadi 기초
python main_wadi_성능test_freq없이_1019.py --cov_type 'spherical' --gamma 0.1 --C 1000.0 \
--exp 'wadi 제대로 돌아가나 테스트중 목표는 0.3 이상 freq데이터 없어'



# ['P1_LIT101', 'P1_MV101', 'P1_P101' ,'P2_P203', 'P3_DPIT301', 'P3_LIT301','P3_MV301','P4_LIT401', 'P5_AIT503']
# 후보군 
# ['P1_FIT101', 'P1_LIT101', 'P1_MV101', 'P1_P101', 'P1_P102', 'P2_AIT201',
#        'P2_AIT202', 'P2_AIT203', 'P2_FIT201', 'P2_MV201', 'P2_P201', 'P2_P202',
#        'P2_P203', 'P2_P204', 'P2_P205', 'P2_P206', 'P3_DPIT301', 'P3_FIT301',
#        'P3_LIT301', 'P3_MV301', 'P3_MV302', 'P3_MV303', 'P3_MV304', 'P3_P301',
#        'P3_P302', 'P4_AIT401', 'P4_AIT402', 'P4_FIT401', 'P4_LIT401',
#        'P4_P401', 'P4_P402', 'P4_P403', 'P4_P404', 'P4_UV401', 'P5_AIT501',
#        'P5_AIT502', 'P5_AIT503', 'P5_AIT504', 'P5_FIT50 1', 'P5_FIT502',
#        'P5_FIT503', 'P5_FIT504', 'P5_P501', 'P5_P502', 'P5_PIT501',
#        'P5_PIT502', 'P5_PIT503', 'P6_FIT601', 'P6_P601', 'P6_P602', 'P6_P603']

# swat
# --freq_select_list 'P1_LIT101'  'P1_MV101'  'P1_P101'  'P2_P203'  'P3_DPIT301'  'P3_LIT301' 'P3_MV301' 'P4_LIT401'  'P5_AIT503'

# wadi
# --freq_select_list '2_FIT_002_PV'  '2_FIT_003_PV'  '2_FQ_101_PV'  '2_FQ_201_PV' \
#        '2_FQ_301_PV'  '2_FQ_401_PV'  '2_FQ_501_PV' '2_FQ_601_PV' \
#        '2_LS_101_AH'  '2_LS_101_AL'  '2_LS_201_AH'  '2_LS_201_AL' \
#        '2_LS_301_AH'  '2_LS_301_AL'


# covariance_type_list = ['full','tied', 'diag', 'spherical']
# gamma_list = [0.000001,  0.1, 1]
# C_list = [0.1, 10.0, 1000.0, 10000.0]