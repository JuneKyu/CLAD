# class_num 43
python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_base' \
  --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 \
  --normal_class_index_list 0 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 --plot_clustering True

python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_base' \
  --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 \
  --normal_class_index_list 1 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 --plot_clustering True

python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_base' \
  --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 \
  --normal_class_index_list 2 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 --plot_clustering True

python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_base' \
  --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 \
  --normal_class_index_list 3 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 --plot_clustering True

python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_base' \
  --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 \
  --normal_class_index_list 4 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 --plot_clustering True

python ../main.py --data_path '../../data' --dataset_name 'gtsrb' --cluster_type 'cvae_base' \
  --dec_pretrain_epochs 100 --dec_train_epochs 100 --classifier 'cnn' --cluster_num 10 --n_hidden_features 300 \
  --normal_class_index_list 5 --temperature 1000 --perturbation 0.01 \
  --classifier_epochs 100 --classifier_lr 0.0001 --plot_clustering True
