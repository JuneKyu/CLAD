# the total number of classes : 43
# 0 : SPDL (Speed Limit)
python ../main.py --data_path '../../data' --dataset_name 'gtsrb' \
  --normal_class_index_list 0 \
  --cluster_type 'cvae_large' --cluster_num 20 --n_hidden_features 50 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 10 --perturbation 0.01 \
  --classifier_epochs 200 --classifier_lr 0.00001 \
  --plot_clustering True --save_cluster_model True
# 1 : INST (Driving Instruction)
python ../main.py --data_path '../../data' --dataset_name 'gtsrb' \
  --normal_class_index_list 1 \
  --cluster_type 'cvae_large' --cluster_num 20 --n_hidden_features 100 \
  --cluster_model_pretrain_epochs 200 --cluster_model_train_epochs 200 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 20 --classifier_lr 0.00001 \
  --plot_clustering True --save_cluster_model True
# 2 : WARN (Warning)
python ../main.py --data_path '../../data' --dataset_name 'gtsrb' \
  --normal_class_index_list 2 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 100 \
  --cluster_model_pretrain_epochs 200 --cluster_model_train_epochs 200 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 200 --classifier_lr 0.001 \
  --plot_clustering True --save_cluster_model True
# 3 : DIRC (Direction)
python ../main.py --data_path '../../data' --dataset_name 'gtsrb' \
  --normal_class_index_list 3 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 300 \
  --cluster_model_pretrain_epochs 200 --cluster_model_train_epochs 200 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 200 --classifier_lr 0.000005 \
  --plot_clustering True --save_cluster_model True
# 4 : SPEC (Special Sign)
python ../main.py --data_path '../../data' --dataset_name 'gtsrb' \
  --normal_class_index_list 4 \
  --cluster_type 'cvae_large' --cluster_num 7 --n_hidden_features 100 \
  --cluster_model_pretrain_epochs 200 --cluster_model_train_epochs 200 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 50 --classifier_lr 0.0001 \
  --plot_clustering True --save_cluster_model True
# 5 : REGN (Regulation)
python ../main.py --data_path '../../data' --dataset_name 'gtsrb' \
  --normal_class_index_list 5 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 300 \
  --cluster_model_pretrain_epochs 200 --cluster_model_train_epochs 200 \
  --classifier_type 'resnet' --temperature 1 --perturbation 0.0001 \
  --classifier_epochs 200 --classifier_lr 0.000005\
  --plot_clustering True --save_cluster_model True

