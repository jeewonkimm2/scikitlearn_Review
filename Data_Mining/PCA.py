#%%

from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# make_regression : 회귀 분석용 가상 데이터 생성
# n_informative : 독립 변수 중 실제로 종속 변수와 상관 관계가 있는 독립 변수의 수(차원)
# effective_rank : 독립 변수 중 서로 독립인 독립 변수의 수 . 만약 None이면 모두 독립
X, y = datasets.make_regression(n_samples = 200, n_features = 10, n_informative = 3, effective_rank = 8)


pca = PCA(n_components = 10)
pca.fit(X)
# mean_ : 평균 벡터(일종의 좌표 원점값)
print(pca.mean_)

# componenets_ : 주성분 벡터
comp = pca.components_
print(comp)

# explained_variance_ : Eigenvalue, 분산값(설명력), PCA 주성분 별로 원본 데이터의 주성분을 얼마나 반영하는지
exp_var = pca.explained_variance_
print(exp_var)

# explained_variance_ratio_ : 각 Eigenvalue의 설명력 percentage. 모든 값의 합은 1에 가까움
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

print(exp_var/exp_var.sum())

cum_exp_var_ratio = np.cumsum(pca.explained_variance_ratio_)

# plt.plot(range(1,11), cum_exp_var_ratio)

Xnew = pca.transform(X)

X0 = X-pca.mean_
np.mean(X0, axis=0)
comp = pca.components_

# .dot : 행렬의 곱
t = np.dot(X0, comp[0])
print(t)
np.var(t)
exp_var[0]


iris = datasets.load_iris()
X2, y2 = iris.data, iris.target

pca = PCA(n_components=3)
pca.fit(X2)
print(pca.explained_variance_)
print(np.cumsum(pca.explained_variance_ratio_))

X2_0 = X2- pca.mean_

T2 = np.matmul(X2_0, pca.components_[:2,:].T)

plt.scatter(T2[:,0],T2[:,1],c=y2)