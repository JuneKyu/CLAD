# rm ../../data/temp_dec -rf

# cifar index = [0 : airplane, 1 : automobile, 2 : bird, 3 : cat, 4 : deer, 5 : dog, 6 : frog, 7 : horse, 8 : ship, 9 : truck]

python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001 --plot_clustering False

# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 8 9 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 10 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 10 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 10 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 30 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 30 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 30 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 50 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 50 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 50 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 100 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 100 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 100 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 3 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 3 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 3 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 5 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 5 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 5 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 8 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 8 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 8 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001


# mnist 0~ 012345678 test code

# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 --temperature 1000 --perterbation 0.00001

# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 7 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 7 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 7 --temperature 1000 --perterbation 0.00001
#
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 7 8 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 7 8 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_temp' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 1 2 3 4 5 6 7 8 --temperature 1000 --perterbation 0.00001


#--------------- best scores for each test -------------------
# cluster_num = 300
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'linear' --cluster_num 5 --n_hidden_features 300 -- normal_class_index_list 0 1 2 --temperature 1 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'fc3' --cluster_num 5 --n_hidden_features 300 -- normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'cnn' --cluster_num 5 --n_hidden_features 300 -- normal_class_index_list 0 1 2 --temperature 1 --perterbation 0.001

# cluster_num = 100
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'linear' --cluster_num 5 --n_hidden_features 100 -- normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'fc3' --cluster_num 5 --n_hidden_features 100 -- normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'cnn' --cluster_num 5 --n_hidden_features 100 -- normal_class_index_list 0 1 2 --temperature 1000 --perterbation 0.001





#
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae' --classifier 'fc3' --cluster_num 5 --n_hidden_features 300 --normal_class_index_list 0 1 2 --perterbation 0.1

# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 10 --normal_class_index_list 0 1 2 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 10 --normal_class_index_list 0 1 2 --perterbation 0.01
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 10 --normal_class_index_list 0 1 2 --perterbation 0.1
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 20 --normal_class_index_list 0 1 2 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 20 --normal_class_index_list 0 1 2 --perterbation 0.01
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 20 --normal_class_index_list 0 1 2 --perterbation 0.1
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 30 --normal_class_index_list 0 1 2 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 30 --normal_class_index_list 0 1 2 --perterbation 0.01
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 30 --normal_class_index_list 0 1 2 --perterbation 0.1
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 100 --normal_class_index_list 0 1 2 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 100 --normal_class_index_list 0 1 2 --perterbation 0.01
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 100 --normal_class_index_list 0 1 2 --perterbation 0.1
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 200 --normal_class_index_list 0 1 2 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 200 --normal_class_index_list 0 1 2 --perterbation 0.01
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 200 --normal_class_index_list 0 1 2 --perterbation 0.1
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 300 --normal_class_index_list 0 1 2 --perterbation 0.001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 300 --normal_class_index_list 0 1 2 --perterbation 0.01
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'cvae' --classifier 'cnn' --cluster_num 5 --n_hidden_features 300 --normal_class_index_list 0 1 2 --perterbation 0.1




# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'cnn' --cluster_num 5 --normal_class_index_list 0
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'cnn' --cluster_num 5 --normal_class_index_list 1
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'cnn' --cluster_num 5 --normal_class_index_list 2
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'cnn' --cluster_num 5 --normal_class_index_list 3
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'cnn' --cluster_num 5 --normal_class_index_list 4
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --classifier 'cnn' --cluster_num 5 --normal_class_index_list 5
#

