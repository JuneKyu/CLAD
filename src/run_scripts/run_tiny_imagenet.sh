# the total number of classes : 40 (out of 200 selected based on Wordnet hierarchy)
# 0 : ANML (Animal)
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' \
  --normal_class_index_list 0\
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.00001 \
  --classifier_epochs 100 --classifier_lr 0.0000001 \
  --plot_clustering True --save_cluster_model True
# 1 : ISCT (Insect)
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' \
  --normal_class_index_list 1\
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.00001 \
  --classifier_epochs 100 --classifier_lr 0.0000001 \
  --plot_clustering True --load_cluster_model True
# 2 : ISTM (Instrument)
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' \
  --normal_class_index_list 2 \
  --cluster_type 'cvae_large' --cluster_num 30 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 300 --cluster_model_train_epochs 300 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.0001 \
  --classifier_epochs 100 --classifier_lr 0.0000005 \
  --plot_clustering True --load_cluster_model True
# 3 : STRT (Structure)
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' \
  --normal_class_index_list 3 \
  --cluster_type 'cvae_large' --cluster_num 10 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.00001 \
  --classifier_epochs 100 --classifier_lr 0.000001 \
  --plot_clustering True --save_cluster_model True
# 4 : VHCL (Viheicle)
python ../main.py --data_path '../../data' --dataset_name 'tiny_imagenet' \
  --normal_class_index_list 4\
  --cluster_type 'cvae_large' --cluster_num 15 --n_hidden_features 200 \
  --cluster_model_pretrain_epochs 100 --cluster_model_train_epochs 100 \
  --classifier_type 'resnet' --temperature 1000 --perturbation 0.00001 \
  --classifier_epochs 100 --classifier_lr 0.0000001 \
  --plot_clustering True --save_cluster_model True



