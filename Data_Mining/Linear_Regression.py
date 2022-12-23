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