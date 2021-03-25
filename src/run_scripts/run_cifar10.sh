python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 0 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 1 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 2 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 3 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 4 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 5 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 6 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 7 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 8 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
  --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 9 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

# 0 1 8 9
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
#   --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 10 --n_hidden_features 200 \
#   --normal_class_index_list 0 1 8 9 --temperature 1000 --perturbation 0.01 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering True --save_cluster_model True

# 2 3 4 5 6 7
# python ../main.py --data_path '../../data' --dataset_name 'cifar10' \
#   --cluster_type 'cvae_large' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 10 --n_hidden_features 100 \
#   --normal_class_index_list 2 3 4 5 6 7 --temperature 1000 --perturbation 0.01 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering True --save_cluster_model True
