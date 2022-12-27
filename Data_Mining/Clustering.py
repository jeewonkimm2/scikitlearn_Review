#%%

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

n = 500

# make_blobs : Generate isotropic Gaussian blobs for clustering. 가우시안 정규분포를 이용한 가상 데이터 생성
x1, y1 = datasets.make_blobs(n_samples=n, random_state=8)

# c : colour
# plt.scatter(x1[:, 0], x1[:,1], c=y1)

x2, y2 = datasets.make_blobs(n_samples=n, random_state=170)
# plt.scatter(x2[:,0], x2[:,1], c=y2)

# data range 바꾸기 [[x축],[y축]]
transformation = [[0.6,-0.6],[-0.4,0.8]]
x2 = np.dot(x2, transformation)
# plt.scatter(x2[:,0],x2[:,1],c=y2)

x3, y3 = datasets.make_blobs(n_samples=n, cluster_std = [1.0,2.0,0.5])
# plt.scatter(x3[:,0], x3[:,1], c=y3)

# 초승달 모양 random data, noise 퍼짐
x4,y4 = datasets.make_moons(n_samples=n, noise = 0.1)
# plt.scatter(x4[:,0], x4[:,1], c=y4)


# AgglomerativeClustering(병합군집) : 시작할 때 각 포인트를 하나의 클러스터로 지정하고, 그다음 종료 조건을 만족할 때까지 가장 비슷한 두 클러스터를 합침
from sklearn.cluster import KMeans, AgglomerativeClustering

kmeans = KMeans(n_clusters = 3)
kmeans.fit(x1)

kmeans_label = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(x1[:, 0], x1[:,1], c=kmeans_label)
