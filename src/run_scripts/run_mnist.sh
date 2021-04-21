# latent class-condition anomaly detection scenario
# curved (0, 3, 8)
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 0 3 8 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 10 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering True --save_cluster_model True
# straight (1, 4, 7)
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 1 4 7 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 10 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering True --save_cluster_model True
# mixed (2 5 6 9)
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 2 5 6 9 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 10 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering True --save_cluster_model True

# single class scenario
# 0
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 0 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 1
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 1 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 2
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 2 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 3
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 4 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 4
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 4 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 5
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 5 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 6
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 6 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 7
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 7 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 8
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 8 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
# 9
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --normal_class_index_list 9 \
  --cluster_type 'cvae' --cluster_num 10 --n_hidden_features 3 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.001 \
  --classifier_epochs 100 --classifier_lr 0.00001 \
  --plot_clustering False --save_cluster_model True
