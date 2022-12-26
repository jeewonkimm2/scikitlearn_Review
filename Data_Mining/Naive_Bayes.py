# 베이즈 추정(Bayesian Estimation)
# 추론 대상의 사전 확률과 추가적인 정보를 기반으로 해당 대상의 사후 확률을 추론하는 통계적 방법
# 특징 : LogisticRegression이나 LinearSVC 같은 선형 분류기보다 훈련 속도가 빠르지만, 일반화 성능은 좀 떨어짐
# 효과적인 이유 : 각 특성을 개별로 취급해 파라미터를 학습하고 각 특성에서 클래스별 통계를 단순하게 취합하기 때문

# (1) GaussianNB : 연속적인 데이터에 적용(매우 고차원인 데이터에 사용), 클래스별로 각 특서의 표준편차와 평균 저장
# (2) BernoulliNB : 이진 데이터에 적용
# (3) MultinomialNB : 카운트 데이터에 적용(특성이 어떤 것을 헤아린 정수 카운트, 예를 들어 문장에 나타난 단어의 횟수), 클래스별로 특성의 평균 계산

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

gausNB = GaussianNB()
gausNB.fit(X,y)

print(gausNB.theta_)
print(gausNB.sigma_)

bernNB = BernoulliNB()

Xmean = np.mean(X, axis=0)
print(Xmean)
Xbin = (X>=Xmean)*1
print(Xbin)

bernNB.fit(Xbin,y)

# feature_log_prob_ : 다항 분포의 각 특징 확률
print(bernNB.feature_log_prob_)
print(np.exp(bernNB.feature_log_prob_))

bernNB2 = BernoulliNB(alpha=2)
bernNB2.fit(Xbin,y)
print(np.exp(bernNB2.feature_log_prob_))


import pandas as pd

# word count
# [5108 rows x 176 columns]
spam = pd.read_csv('https://drive.google.com/uc?export=download&id=1l6gUFvs4PNoY2OVg44hCNmOREfEsx2qX')
print(spam)

multiNB = MultinomialNB()

X = spam.drop('target',axis=1)
y = spam['target']
multiNB.fit(X,y)
print(multiNB.feature_log_prob_)
print(multiNB.feature_log_prob_.shape)
print(np.exp(multiNB.feature_log_prob_))
print(np.sum(np.exp(multiNB.feature_log_prob_),axis=1))


from sklearn.naive_bayes import CategoricalNB
import pandas as pd

car = pd.read_csv('https://drive.google.com/uc?export=download&id=1wFAkAmsIBiLQXXejiFVJr_TqUbm9HMDP')
# 'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'
print(car.columns)
print(car)

print(car['buying'].unique())
print(car['doors'].unique())
print(car['persons'].unique())
print(car['class'].unique())

categNB = CategoricalNB()
categNB.fit(X,y)

# LabelEncoder : Category형 Feature를 숫자로 변환하는 것을 의미한다. 예컨대 iris datasets에서 target 값인 ['setosa' 'versicolor' 'virginica']를 [0, 1, 2]로 변환하는 방식이다.
# 그러나 Label encoding의 경우 숫자의 크고 적음이 머신러닝 학습에 영향을 끼칠 수 있다. 즉 단순 정수 값이 순서 또는 중요도로 작용할 수 있다. 따라서 선형회귀 등의 알고리즘에는 적용하면 안 된다. 반대로 Tree 계열의 알고리즘은 숫자의 특성을 반영하지 않으므로 Label encoding을 해도 학습에 영향이 없다.
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

enc.fit_transform(['a','b','c','a'])

Xenc = X.copy()

for col in X.columns:
    Xenc[col] = enc.fit_transform(X[col])

categNB.fit(Xenc,y)

print(categNB.feature_log_prob_)
print(len(categNB.feature_log_prob_))

print(categNB.feature_log_prob_[3].shape)

print(car['buying'].unique())
print(car['persons'].unique())


import numpy as np
np.exp(categNB.feature_log_prob_[3])
np.sum(np.exp(categNB.feature_log_prob_[3]),axis=1)