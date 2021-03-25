# single class scenario

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 0 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 1 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 2 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 3 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 4 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 5 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 6 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 7 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 8 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 3 \
  --normal_class_index_list 9 --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
