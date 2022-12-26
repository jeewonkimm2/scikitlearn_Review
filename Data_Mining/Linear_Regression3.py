#%%

from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

data = datasets.load_boston()

X = data.data
y = data.target

reg = LinearRegression()
reg.fit(X,y)
y_pred = reg.predict(X)
errors = y - y_pred
print(np.mean(errors))

# plt.hist(errors, bins = 20)

# QQ plot
# stats.probplot(errors, dist = "norm", plot = plt)
# help(stats.probplot)

# skewness 치우처짐 정도
S = stats.skew(errors)
stats.kurtosis(errors, fisher=True)
# Kurtosis, thickness of tail 그래프 꼬리의 두께
C = stats.kurtosis(errors, fisher=False)

# k : the number of input variables
k = 13

# Jarque-Bera test : Goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution
JB = (len(X)-k)/6*(S**2 + 0.25 * (C-3)**2)

alpha = 0.05
stats.chi2.ppf(1-alpha,2)
1-stats.chi2.cdf(JB,2)

plt.scatter(X[:,1],errors)

# Breusch-Pagen Test, performing auxiliary regression
errors2 = errors**2
reg.fit(X, errors2)
pred_errors2 = reg.predict(X)

SSE = np.sum((errors2-pred_errors2)**2)
SSR = np.sum((pred_errors2-np.mean(errors2))**2)

MSE = SSE/(len(X)-k-1)
MSR = SSR/k

f = MSR/MSE

stats.f.ppf(1-alpha, k, len(X)-k-1)
1-stats.f.cdf(f, k, len(X)-k-1)

r2 = reg.score(X, errors2)
print("R2 : ", r2)
LM = len(X)*r2
print("LM : ", LM)

stats.chi2.ppf(1-alpha, k)
1-stats.chi2.cdf(LM,k)