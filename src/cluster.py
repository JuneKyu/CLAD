from sklearn.cluster import *
from sklearn.mixture import *

def make_arg_cluster(dict_):
    str_ = '('
    i = 0
    for key, values in dict_.items():
        str_ += key + '=' + str(values)
        if i != 1:
            str_ += ', '
        i += 1

    str_ = str_ + ')'

    return str_



# 클러스터 함수를 통해 나오는 아웃풋 : 인풋 데이터와 그 인풋 데이터가 클러스터로 라벨링 된 값.
def KMeans_clustering(**kwargs):
#     TODO
# cluster_args 에서 다양한 아규먼트중 해당 모델에 맞게 맞춰야 할듯? 

    model = KMeans(n_clusters=kwargs['n_clusters'])
    model.fit(kwargs['train_x'])
    train_label = model.labels_
    model.fit(kwargs['val_x'])
    val_label = model.labels_
    model.fit(kwargs['test_x'])
    test_label = model.labels_
    
    return train_label, val_label, test_label, model

def MeanShift_clustering(**kwargs):
    bandwidth = estimate_bandwidth(kwargs['train_x'], quantile=0.2, n_samples=500)
    model = MeanShift(bandwidth = bandwidth, bin_seeding = True)
    model.fit(kwargs['train_x'])
    train_label = model.labels_
    model.fit(kwargs['val_x'])
    val_label = model.labels_
    model.fit(kwargs['test_x'])
    test_label = model.labels_
    
    return train_label, val_label, test_label, model


def DBSCAN_clustering(**kwargs):
    model = DBSCAN(eps = kwargs['eps'], min_samples = kwargs['min_samples'])
#     질문?
#     core_samples_mask = np.zeros_like(cluster_model.labels_, dtype=bool)
#     core_samples_mask[cluster_model.core_sample_indices_] = True

#     labels = cluster_model.labels_
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#     print('Estimated number of clusters : %d' % n_clusters_)
#     print('Estimated number of noises : %d' % n_noise_)
    model.fit(kwargs['train_x'])
    train_label = model.labels_
    model.fit(kwargs['val_x'])
    val_label = model.labels_
    model.fit(kwargs['test_x'])
    test_label = model.labels_
    
    return train_label, val_label, test_label, model

# return labels from GaussianMixture
def GaussianMixture_clustering(**kwargs):
    model = GaussianMixture(n_components = kwargs['n_clusters'], covariance_type = kwargs['covariance_type'])
    
    model.fit(kwargs['train_x'])
    train_label = model.predict(kwargs['train_x'])
    model.fit(kwargs['val_x'])
    val_label = model.predict(kwargs['val_x'])
    model.fit(kwargs['test_x'])
    test_label = model.predict(kwargs['test_x'])

    return train_label, val_label, test_label, model
