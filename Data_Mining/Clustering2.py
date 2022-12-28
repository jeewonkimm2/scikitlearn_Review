#%%

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
# AgglomerativeClustering(병합군집) : 시작할 때 각 포인트를 하나의 클러스터로 지정하고, 그다음 종료 조건을 만족할 때까지 가장 비슷한 두 클러스터를 합침
from sklearn.cluster import KMeans, AgglomerativeClustering, k_means

n=500
x1, y1 = datasets.make_blobs(n_samples=n, random_state=8)
x2, y2 = datasets.make_blobs(n_samples=n, random_state=170)
x4, y4 = datasets.make_moons(n_samples=n, noise = 0.1)

aggl = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'complete')
aggl.fit(x1)
aggl_label1 = aggl.labels_
# plt.scatter(x1[:,0], x1[:,1], c=aggl_label1)

aggl = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage='ward')
aggl.fit(x2)
aggl_label2 = aggl.labels_
# plt.scatter(x2[:,0], x2[:,1], c=aggl_label2)

aggl = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'complete')
aggl.fit(x4)
aggl_label4 = aggl.labels_
# plt.scatter(x4[:,0], x4[:,1], c=aggl_label4)


x = [[30,10],[26,11],[16,16],[20,17],[19,18]]
aggl = AgglomerativeClustering(n_clusters=1, affinity = 'euclidean', linkage = 'complete')
aggl.fit(x)
# children_ : Hierarchical Clustering의 자식 노드
print(aggl.children_)


from scipy.cluster import hierarchy

Z = hierarchy.linkage(x1, method = 'ward', metric = 'euclidean')
scipy_hierarchy_label = hierarchy.cut_tree(Z, n_clusters=[3,5,7])
print(scipy_hierarchy_label.shape)
np.max(scipy_hierarchy_label, axis=0)

fig = plt.figure(figsize=(10,6))
# dendrogram : 클러스터링의 결과를 시각화하기 위한 대표적인 그래프, 대표적으로 hierarchical clustering 방식에 대해 시각화하는 그래프로 많이 활용됨
hierarchy.dendrogram(Z)


# silhouette_score(실루엣 스코어) : 타겟 없을때 사용, 전체 실루엣 계수의 평균값, -1~1 사이의 값을 가짐. 군집간 거리는 멀고 군집내 거리는 가까울수록 점수가 높다. 0.5보다 크면 클러스터링이 잘 된 거라 평가함.
# homogeneity_score : 각 군집(예측값)이 동일한 클래스(실제값)로 구성되어 있는 정도
# completeness_score : 각 클래스(실제값)에 대하여 동일한 군집(예측값)으로 구성되는 정도
# adjusted_rand_score : 타겟을 사용하여 실제 정답과 비교하여 군집 평가
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, adjusted_rand_score