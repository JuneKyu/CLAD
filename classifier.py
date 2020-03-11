from sklearn.neighbors import KNeighborsClassifier

def KNN_classifier(**kwargs):
    model = KNeighborsClassifier(n_neighbors=kwargs['n_neighbors'])

    model.fit(kwargs['train_x'], kwargs['label_for_train'])
    
    return model