# latent class-condition anomaly detection scenario
# things (0 1 8 9)
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 0 1 8 9 \ 
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering True --save_cluster_model True
# living (2 3 4 5 6 7)
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 2 3 4 5 6 7 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 100 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering True --save_cluster_model True

# single class scenario
# 0
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 0 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 1
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 1 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 2
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 2 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 3
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 3 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 4
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 4 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 5
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 5 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 6
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 6 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 7
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 7 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 8
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 8 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
# 9
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --normal_class_index_list 9 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

