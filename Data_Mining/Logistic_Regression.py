import numpy as np
from scipy import stats

# likelihood function when normal distribution
def norm(x, mu = 0, sigma = 1):
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/2/sigma**2)

D = [2.61, 3.73, 2.80, 4.29, 3.12]

# np.prod : array 내부의 곱
np.prod([1,2,3,4])

np.prod([norm(x, mu=-15) for x in D])
np.sum([np.log(norm(x, mu=-15)) for x in D])



from scipy.optimize import minimize

# Log likelihood function(generalization)
def logL(x,data,sigma):
    return np.sum([-np.log(norm(d,mu=x,sigma=sigma)) for d in data])

# 최적화 값 찾기, x가 3.31일때
res = minimize(logL, 0 ,(D,1), method = 'Nelder-Mead', options = {'xatol':1e-8, 'disp':True})
print(res)
# 이 값이 3.31이다
print(np.mean(D))
# 결론적으로 res == np.mean(D)



from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pandas as pd
import sklearn

# 92 rows, 2 columns(height, sex)
height = pd.read_csv('https://drive.google.com/uc?export=download&id=1m0noi5t5StwPdTACZOkP22hKKR6v4vLX')
print(height)
print(height.shape)

# L2 Regularization(0으로 근사)사용. C값이 낮을수록 계수를 0으로 근사하는 regularization이 강화됨
# Regularization C값이 클수록 decision boundary는 최대한 train data를 맞추려고 함. 즉 overfitting의 가능성이 증가합니다.
clf = LogisticRegression(C=10)

# LogisticRegression model training
clf.fit(height[['height']],height['sex'])
print(clf.coef_)
print(clf.intercept_)

clf.predict(height[['height']])

# predict_proba : 각 클래스에 대한 확률
y_prob = clf.predict_proba(height[['height']])
print(y_prob)

# axis=1은 y축을 기준으로 row 별로 존재하는 column들의 값을 합쳐 1개로 축소하는 과정
# 각 확률의 합이 1이란 뜻
print(np.sum(y_prob,axis=1))


iris = datasets.load_iris()

X = iris.data
y = iris.target

clf2 = LogisticRegression(C=10e5)
clf2.fit(X,y)

print(clf2.coef_)
print(clf2.intercept_)

y_pred2 = clf2.predict(X)
y_prod2 = clf2.predict_proba(X)
print(y_prod2)
print(np.sum(y_prod2,axis=1))

print(clf2.score(X,y))

print(np.sum(y_pred2==y)/len(X))
