#%%

import pandas as pd
import numpy as np
from scipy import stats
from numpy import linalg

petrol = pd.read_csv('/Users/jeewonkim/Desktop/data/petrol.csv')
print(petrol.columns)

n = len(petrol)

X = petrol[['tax','income','highway','license']].values
X = np.c_[np.ones(n),X]

XtX = np.matmul(X.T, X)
print(XtX.shape)

inv_XtX = linalg.inv(XtX)

y = petrol[['consumption']].values

# Test Concerning Regression Coefficients(T-test)
# T-test는 each Beta를 위한다

beta = np.matmul(np.matmul(inv_XtX, X.T),y)
print(beta)

# .flatten() == .reshape(-1) == .ravel() : making 1 dimension
beta = beta.flatten()
# [ 3.77291146e+02 -3.47901492e+01 -6.65887518e-02 -2.42588889e-03
print(beta)

# [[ 3.77291146e+02]
#  [-3.47901492e+01]
#  [-6.65887518e-02]
#  [-2.42588889e-03]
#  [ 1.33644936e+03]]
print(beta.reshape((-1,1)))


y_pred = np.matmul(X, beta.reshape((-1,1)))
y_pred = y_pred.flatten()
y_pred

SSE = np.sum((y.flatten()-y_pred)**2)

# X.shape[1] == 5
MSE = SSE/(n-X.shape[1])

cov_beta = MSE * inv_XtX
print(cov_beta)

# 1 4 6이 대각성 성분인 대각행렬, Diagonal Matrix
# [[1 0 0]
#  [0 4 0]
#  [0 0 6]]
print(np.diag([1,4,6]))

# sqrt : 제곱근 square root
se_beta = np.sqrt(np.diag(cov_beta))
print(se_beta)

t = beta[1]/se_beta[1]
print("t : ", t)

alpha = 0.05
# H0 : Beta = 0
H0 = stats.t.ppf(1-alpha/2,n-X.shape[1])
print(H0)
# H1 : Beta != 0
H1 = (1-stats.t.cdf(np.abs(t),n-X.shape[1])*2)
print(H1)

if(t <= H0 and t>= -H0):
    print("H0 : Beta == 0")
else:
    print("H1 : Beta != 0")
    
    


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

reg = LinearRegression()

reg.fit(petrol[['tax','income','highway','license']],petrol['consumption'])

# R2 : Statistical measures for Goodness-of-fit
r2 = reg.score(petrol[['tax','income','highway','license']],petrol['consumption'])
print("r2 : ", r2)

n = len(petrol)
p = 4

adj_r2 = 1 - (n-1)/(n-p-1) * (1-r2)
print("Adjusted R2 : ", adj_r2)


# 컬럼을 하나씩 추가하면서 성능 관찰
r2 = []
adj_r2 = []
cols = ['license','tax','income','highway']

for i in range(len(cols)):
    reg.fit(petrol[cols[:i+1]], petrol['consumption'])
    r2.append(reg.score(petrol[cols[:i+1]],petrol['consumption']))
    adj_r2.append(1-(n-1)/(n-(i+1)-1)*(1-r2[-1]))



plt.plot(range(1,len(cols)+1), r2, label = 'r2')
plt.plot(range(1,len(cols)+1),adj_r2, label = "adj r2")
plt.legend()

cov_mat = petrol[['tax','income','highway','license']].cov()
corr_mat = petrol[['tax','income','highway','license']].corr()


reg.fit(petrol[['income','highway','license']], petrol['tax'])
r2 = reg.score(petrol[['income','highway','license']], petrol['tax'])
print(r2)

# VIF(Variance Inflation Factors) 분산 팽창 요인 : 다중 회귀 모델에서 독립 변수간 상관 관계가 있는지 측정하는 척도
# If VIF > 10 then multicollinearity(다중공선성) is high (10이상이면 쓰지 않기!)
vif = 1/(1-r2)
print("VIF : ", vif)