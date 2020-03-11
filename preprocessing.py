import sklearn.preprocessing
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
def PCA_preprocessing_modify(**kwargs):
    if kwargs['scaler'] == 'standard':
        transformer = sklearn.preprocessing.StandardScaler()
        transformer.fit(kwargs['train_x'])
    elif kwargs['scaler'] == 'normalizer':
        transformer = sklearn.preprocessing.Normalizer()
        transformer.fit(kwargs['train_x'])

    train_x = transformer.transform(kwargs['train_x'])
    val_x = transformer.transform(kwargs['val_x'])
    test_x = transformer.transform(kwargs['test_x'])

    pca = PCA(n_components = np.shape(train_x)[1])
    pca.fit(train_x)

    train_x = pca.transform(train_x)[:,kwargs['selected_dim']]
    val_x = pca.transform(val_x)[:, kwargs['selected_dim']]
    test_x = pca.transform(test_x)[:, kwargs['selected_dim']]

    return train_x, val_x, test_x

def PCA_preprocessing_modify_2(**kwargs):
    if kwargs['scaler'] == 'standard':
        transformer = sklearn.preprocessing.StandardScaler()
        transformer.fit(kwargs['train_x'])
    elif kwargs['scaler'] == 'normalizer':
        transformer = sklearn.preprocessing.Normalizer()
        transformer.fit(kwargs['train_x'])

    train_x = transformer.transform(kwargs['train_x'])
    val_x = transformer.transform(kwargs['val_x'])
    test_x = transformer.transform(kwargs['test_x'])

    pca = PCA(n_components = np.shape(train_x)[1])
    pca.fit(train_x)

    train_x = pca.transform(train_x)[:,kwargs['selected_dim']]
    val_x = pca.transform(val_x)[:, kwargs['selected_dim']]
    test_x = pca.transform(test_x)[:, kwargs['selected_dim']]

    return train_x, val_x, test_x,pca


def PCA_preprocessing(**kwargs):
    if kwargs['scaler'] == 'standard':
        transformer = sklearn.preprocessing.StandardScaler()
        transformer.fit(kwargs['train_x'])
    elif kwargs['scaler'] == 'normalizer':
        transformer = sklearn.preprocessing.Normalizer()
        transformer.fit(kwargs['train_x'])

    train_x = transformer.transform(kwargs['train_x'])
    val_x = transformer.transform(kwargs['val_x'])
    test_x = transformer.transform(kwargs['test_x'])
    feature_num = kwargs['feature_num']

    pca = PCA(n_components = np.shape(train_x)[1])
    pca.fit(train_x)

    train_x = pca.transform(train_x)[:, 0:feature_num]
    val_x = pca.transform(val_x)[:, 0:feature_num]
    test_x = pca.transform(test_x)[:, 0:feature_num]

    return train_x, val_x, test_x


def PCAkernel_preprocessing(**kwargs):
    if kwargs['scaler'] == 'standard':
        transformer = sklearn.preprocessing.StandardScaler()
        transformer.fit(kwargs['train_x'])
    elif kwargs['scaler'] == 'normalizer':
        transformer = sklearn.preprocessing.Normalizer()
        transformer.fit(kwargs['train_x'])

    train_x = transformer.transform(kwargs['train_x'])
    val_x = transformer.transform(kwargs['val_x'])
    test_x = transformer.transform(kwargs['test_x'])
    feature_num = kwargs['feature_num']

    pca = KernelPCA(n_components = np.shape(train_x)[1], kernel = kwargs['kernel'])
    pca.fit(train_x)

    train_x = pca.transform(train_x)[:, 0:feature_num]
    val_x = pca.transform(val_x)[:, 0:feature_num]
    test_x = pca.transform(test_x)[:, 0:feature_num]

    return train_x, val_x, test_x


def LLE_preprocessing(**kwargs):
    if kwargs['scaler'] == 'standard':
        transformer = sklearn.preprocessing.StandardScaler()
        transformer.fit(kwargs['train_x'])
    elif kwargs['scaler'] == 'normalizer':
        transformer = sklearn.preprocessing.Normalizer()
        transformer.fit(kwargs['train_x'])
        
    train_x = transformer.transform(kwargs['train_x'])
    val_x = transformer.transform(kwargs['val_x'])
    test_x = transformer.transform(kwargs['test_x'])
    
    lle = LocallyLinearEmbedding(
        n_neighbors = kwargs['n_neighbors'],
        n_components = kwargs['n_components'],
        eigen_solver = 'auto',
        method = kwargs['method']
    )
    lle.fit(train_x)
    train_x = lle.transform(train_x)
    val_x = lle.transform(val_x)
    test_x = lle.transform(test_x)

    return train_x, val_x, test_x
        
