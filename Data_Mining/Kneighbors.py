from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

print(len(X_train))
print(len(X_test))

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

y_pred = knn_clf.predict(X_test)
print(y_pred)

# .kneighbors : 인접한 k개의 sample에 대해 거리, index 반환
nn = knn_clf.kneighbors(X_test)
print(nn)

# Manhattan Distance
knn_clf = KNeighborsClassifier(n_neighbors = 3, metric = 'manhattan')
knn_clf.fit(X_train, y_train)
y_pred2 = knn_clf.predict(X_test)
print(y_pred2)


import numpy as np

# 각 모델이 몇개나 같은 값으로 예측했는지
print(np.sum(y_pred == y_pred2))

cov_mat = np.cov(X_train.T)
print(cov_mat.shape)

# Mahalanobis Distance
# metric_params : Additional keyword arguments for the metric function.
knn_clf = KNeighborsClassifier(n_neighbors = 5, metric = 'mahalanobis', metric_params={'V':cov_mat})
knn_clf.fit(X_train, y_train)
y_pred3 = knn_clf.predict(X_test)
print(y_pred3)
nn3 = knn_clf.kneighbors(X_test)
print(nn3)
# Check score
print(knn_clf.score(X_test, y_test))

rnn_clf = RadiusNeighborsClassifier(radius = 0.7, weights = 'distance')
rnn_clf.fit(X_train, y_train)
# y_pred4 = rnn_clf.predict(X_test)
# print(y_pred4)
nn4 = rnn_clf.radius_neighbors(X_test)
print(nn4)

n_neighbors = [len(x) for x in nn4[1]]
print(n_neighbors)