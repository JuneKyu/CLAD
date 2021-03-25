# mnist 0 3 8 cluster num
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 2 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 4 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 6 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 8 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 10 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 12 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 14 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 16 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 18 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True
#
# python ../main.py --data_path '../../data' --dataset_name 'mnist' \
#   --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
#   --classifier 'resnet' --cluster_num 20 --n_hidden_features 10 \
#   --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 100 --classifier_lr 0.0001 \
#   --plot_clustering False --save_cluster_model True

# mnist 0 3 8 n_hidden_features 
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 10 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 20 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 30 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 40 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 50 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 60 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 70 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 80 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 90 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 100 \
  --normal_class_index_list 0 3 8 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

# mnist 1 4 7 cluster_num

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 2 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 4 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 6 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 8 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True \

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 12 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 14 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 16 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 18 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 20 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True


# mnist 1 4 7 n_hidden_features
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 10 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 20 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 30 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 40 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 50 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 60 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 70 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 80 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 90 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 100 \
  --normal_class_index_list 1 4 7 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True


# mnist 2 5 6 9
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 2 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 4 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 6 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 8 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 12 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 14 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 16 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True
 
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 18 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 20 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

# mnist 2 5 6 9 n_hidden_features
python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 10 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 20 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 30 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 40 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 50 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 60 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 70 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 80 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 90 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

python ../main.py --data_path '../../data' --dataset_name 'mnist' \
  --cluster_type 'cvae_base' --dec_pretrain_epochs 100 --dec_train_epochs 100 \
  --classifier 'resnet' --cluster_num 10 --n_hidden_features 100 \
  --normal_class_index_list 2 5 6 9 --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0001 \
  --plot_clustering False --save_cluster_model True

