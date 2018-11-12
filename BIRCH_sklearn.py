import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch
from sklearn import metrics
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3],
                  random_state =9)
if __name__ == '__main__':
    print("unclustered:")
    print(X)
    y_pred = Birch(n_clusters=None).fit_predict(X)
    print("result:")
    print(y_pred)
    print("Calinski-Harabasz Score:", metrics.calinski_harabaz_score(X, y_pred))