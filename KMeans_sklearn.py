import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import metrics
"""
在普通的K Means的计算过程中，
每次更新各聚类中心点时，需要计算所有点和每个聚类中心点的距离，所以代价特别昂贵。
而在mini-batch K Means的计算过程中，每次更新各聚类中心点时，
先从所有数据中随机地选取一个小集合（也就是这里的mini-batch），根据这个集合中的数据点，
来更新各聚类的中心点。
下一次更新时，再重新从所有数据点中选取一个随机的小集合，如此重复，直到达到收敛条件。
"""
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state =9)
if __name__ == '__main__':
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    print('kmeans_calinski_harabaz_score:',metrics.calinski_harabaz_score(X, y_pred))
    y_pred = MiniBatchKMeans(n_clusters=2, batch_size=200, random_state=9).fit_predict(X)
    score = metrics.calinski_harabaz_score(X, y_pred)
    print('MiniBatchKMeans_calinski_harabaz_score:',metrics.calinski_harabaz_score(X,y_pred))
