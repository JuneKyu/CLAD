# rm ../../data/temp_dec -rf
# 0 1 8 9
python ../main.py --data_path '../../data' --dataset_name 'cifar10' --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 0 1 8 9 --temperature 1000 --perturbation 0.01 --classifier_epochs 100 --classifier_lr 0.0001 --plot_clustering True
