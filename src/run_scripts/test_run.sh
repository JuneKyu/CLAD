# rm ../../data/temp_dec -rf

# cifar index = [0 : airplane, 1 : automobile, 2 : bird, 3 : cat, 4 : deer, 5 : dog, 6 : frog, 7 : horse, 8 : ship, 9 : truck]
# swat normal 5600~
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 5 --normal_class_index_list 0 --temperature 1000 --perturbation 0.001 --plot_clustering False --classifier_epochs 5000 --classifier_lr 0.0001



# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 5 --n_hidden_features 300 --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 300 --classifier_lr 0.00001 # 8645ㅇ

# cifar10 single class test script

# mnist 0~ 012345678 test code


