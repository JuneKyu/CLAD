# set ready
# 0
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' --cluster_type 'cvae_large' \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 --classifier_type 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 0\
  --temperature 1000 --perturbation 0.00001 --classifier_epochs 100 \
  --classifier_lr 0.0000001 --plot_clustering True --save_cluster_model True
# 1
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' --cluster_type 'cvae_large' \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 --classifier_type 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 1\
  --temperature 1000 --perturbation 0.00001 --classifier_epochs 100 \
  --classifier_lr 0.0000001 --plot_clustering True --load_cluster_model True
# 2
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' --cluster_type 'cvae_large' \
  --cluster_model_pretrain_epochs 300 --cluster_model_train_epochs 300 --classifier_type 'resnet' --cluster_num 30 --n_hidden_features 200 \
  --normal_class_index_list 2\
  --temperature 1000 --perturbation 0.0001 --classifier_epochs 100 \
  --classifier_lr 0.0000005 --plot_clustering True --load_cluster_model True
# 3
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' --cluster_type 'cvae_large' \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 --classifier_type 'resnet' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 3\
  --temperature 1000 --perturbation 0.00001 --classifier_epochs 100 \
  --classifier_lr 0.000001 --plot_clustering True --save_cluster_model True
# 4
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' --cluster_type 'cvae_large' \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 --classifier_type 'resnet' --cluster_num 15 --n_hidden_features 200 \
  --normal_class_index_list 4\
  --temperature 1000 --perturbation 0.00001 --classifier_epochs 100 \
  --classifier_lr 0.0000001 --plot_clustering True --save_cluster_model True



