# mnist 0 3 8
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 3 --n_hidden_features 300 --normal_class_index_list 0 3 8 --temperature 1 --perturbation 0.001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.001 --plot_clustering True --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 3 --n_hidden_features 300 --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.0001 --plot_clustering True --save_cluster_model True
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 3 --n_hidden_features 300 --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 5000 --classifier_lr 0.0001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 3 --n_hidden_features 300 --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.00001 --save_cluster_model True
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 3 --n_hidden_features 300 --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.00001 --load_cluster_model True


# mnist 2 5 6 9
# rm ../../data/temp_dec -rf
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 5000 --classifier_lr 0.0001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.00001
# rm ../../data/temp_dec -rf


# mnist 1 4 7
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'linear' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 5000 --classifier_lr 0.0001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'fc3' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.00001
# python ../main.py --data_path '../../data' --dataset_name 'mnist' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 5 --n_hidden_features 300 --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 300 --classifier_lr 0.00001 # 8670


