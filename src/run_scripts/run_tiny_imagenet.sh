
# python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 0 1 2 3 4 5 6 7 8 9 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.001 --plot_clustering True --save_cluster_model True
# python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' --cluster_type 'dec' --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 --normal_class_index_list 0 1 2 3 4 5 6 7 8 9 --temperature 1000 --perturbation 0.00001 --plot_clustering False --classifier_epochs 100 --classifier_lr 0.001 --plot_clustering True --load_cluster_model True


python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' --cluster_type 'cvae_base' \
  --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 200 \
  --normal_class_index_list 0\
  --temperature 1000 --perturbation 0.001 --plot_clustering False --classifier_epochs 100 \
  --classifier_lr 0.00001 --plot_clustering True --load_cluster_model True

# animal
# cluster_num = 40 ; 50.19 / 50.50
# cluster_num = 30 ; 52.12 / 50.02 
# cluster_num = 20 ; 50.63 / 51.70
# cluster_num = 10 ; 51.02 / 47.07

# classifier_lr ->
# 0.0001 ; 51.02 / 50.32
# 0.001 ; 49.17 / 45.11 
# 0.00001

