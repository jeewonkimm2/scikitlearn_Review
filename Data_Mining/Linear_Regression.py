#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Importing ice cream sales data
# 12 rows, 2 columns(temperature, sales)
sales = pd.read_csv('http://drive.google.com/uc?export=download&id=1n9SDdK2pFbM0H14ZSRLB8HpreFYBl6KH')

plt.scatter(sales['temperature'],sales['sales'])

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
# Data Training : .fit
reg.fit(sales[['temperature']],sales['sales'])

sales['temperature'].shape
sales[['temperature']].shape

reg.coef_
reg.intercept_

# Predicting ice cream sales using trained model
y_pred = reg.predict(sales[['temperature']])
# Predicting sales of random temperature value
reg.predict([[13.1],[20.7]])

y_true = sales['sales'].values

# SST : Sum of squares total (SSE+SSR)
# 실제값과 Y들의 평균값과의 차이, 즉 Y의 평균으로 예측했을 때 얼마나 정확할까를 나타냄
SST = np.sum((y_true-np.mean(y_true))**2)
print("SST : ", SST)

# SSR : Regression sum of squares
# 선형회귀모형의 예측값과 Y들의 평균값과의 차이, 선형회귀모형으로 예측했을 때와, Y의 평균으로 예측했을 때의 차이
SSR = np.sum((y_pred-np.mean(y_true))**2)
print("SSR : ", SSR)

# SSE : Error sum of squares
# 실제값과 선형회귀모형의 예측값과의 차이, 즉 선형회귀모형으로 예측했을 때 얼마나 정확할까를 나타냄
SSE = np.sum((y_true-y_pred)**2)
print("SSE : ", SSE)

print(SSE+SSR)
print(SST)


# 단순회귀모델: 회귀선의 적합도 평가

p = 1
n = len(sales)

MSR = SSR/p
MSE = SSE/(n-p-1)

# F-Test
ftest = MSR/MSE
print("F-Test : ", ftest)

# ppf : Percent point function (inverse of cdf) at 0.95
print(stats.f.ppf(0.95, p, n-p-1))
# cdf : Cumulative distribution function of F
print(1-stats.f.cdf(ftest,p,n-p-1))



# Importing petrol data
# 49 rows, 5 columns('tax','income','highway','driver','petrol')
petrol = pd.read_csv('/Users/jeewonkim/Desktop/petrol.csv')
print(petrol.head())
print(petrol.dtypes)



# Data Training
reg.fit(petrol[['tax','income','highway','license']],petrol['consumption'])

print("Coefficients of petrol :", reg.coef_)
print("Intercept of petrol :", reg.intercept_)

from numpy import linalg
X = petrol[['tax','income','highway','license']].values

# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
print(np.ones((13,)))
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]]
print(np.ones((12,2)))

# 가로 방향으로 두 배열 합치기
# 첫번째 column에 1추가 후 X와 합침
X1 = np.c_[np.ones(len(X)),X]
print(X1)
print(X1.shape, X1.T.shape)


A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(A*B)

# X1 transpose * X1
XtX = np.matmul(X1.T, X1)
print(XtX.shape)

inv_XtX = linalg.inv(XtX)
I = np.matmul(XtX, inv_XtX)

# 한꺼번에 intercept와 coefficients구하는 방법
beta = np.matmul(np.matmul(inv_XtX,X1.T),petrol[['consumption']].values)
# 숫자가 동일한지 확인하는 과정
print(beta[0], reg.intercept_)
print(beta[1], reg.coef_[0])

y_true = petrol['consumption'].values
y_pred = reg.predict(petrol[['tax','income','highway','license']])

SST = np.sum((y_true-np.mean(y_true))**2)
SSR = np.sum((y_pred-np.mean(y_true))**2)
SSE = np.sum((y_true-y_pred)**2)

print(SST, SSR+SSE)

p = 4
n = petrol.shape[0]

MSR = SSR/p
MSE = SSE/(n-p-1)
print(MSR, MSE)

ftest = MSR/MSE
print(ftest)

from scipy import stats
print(stats.f.ppf(0.95, p, n-p-1))
print(1-stats.f.cdf(ftest,p,n-p-1))
