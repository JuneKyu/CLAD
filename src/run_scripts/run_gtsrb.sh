# class_num 43


# set ready
# 3
# python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_large' \
#   --dec_pretrain_epochs 200 --dec_train_epochs 200 --classifier 'resnet' --cluster_num 10 --n_hidden_features 300 \
#   --normal_class_index_list 3 --temperature 1000 --perturbation 0.0001 \
#   --classifier_epochs 200 --classifier_lr 0.000005 --plot_clustering True --save_cluster_model True

# 5
# 79.05
# python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_large' \
#   --dec_pretrain_epochs 200 --dec_train_epochs 200 --classifier 'resnet' --cluster_num 10 --n_hidden_features 300 \
#   --normal_class_index_list 5 --temperature 1 --perturbation 0.0001 \
#   --classifier_epochs 200 --classifier_lr 0.000005 --plot_clustering True --save_cluster_model True



python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_large' \
  --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'resnet' --cluster_num 20 --n_hidden_features 50 \
  --normal_class_index_list 0 --temperature 10 --perturbation 0.01 \
  --classifier_epochs 200 --classifier_lr 0.00001 --plot_clustering False --save_cluster_model True

