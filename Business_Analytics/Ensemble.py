import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# RandomForest Classifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 0)

# n_estimators : The number of trees
# n_estimators가 많을수록 안정된 모델. 100~200 -> enough
# **Hard voting** : Predict class label as the class that represents the majority (mode) of the class labels predicted by each individual classifier.    
forest = RandomForestClassifier(n_estimators = 5, random_state = 0)
forest.fit(X_train, y_train)
y_train_hat = forest.predict(X_test)

print(forest.estimators_[0].predict(X_test))
print(forest.estimators_)
print(forest.feature_importances_)


# Building random forest with decision tree classifier
from sklearn.tree import DecisionTreeClassifier
n_estimators = 100
random_trees = []

for i in range(n_estimators):
    idx = np.random.choice(X_train.shape[0], X_train.shape[0], replace = True)
    
    X_train_base = X_train[idx, :]
    y_train_base = y_train[idx]

    dt = DecisionTreeClassifier(max_features = 'sqrt')
    dt.fit(X_train_base, y_train_base)
    random_trees.append(dt)
    
y_test_hats = []
for i in range(n_estimators):
    y_test_hats.append(random_trees[i].predict(X_test))

y_test_hats = np.stack(y_test_hats).T

print(y_test_hats.shape)

from scipy import stats
# .mode : 최빈값
# .squeeze() : 1인 차원 제거
y_test_hat_voted = stats.mode(y_test_hats, axis=1)[0].squeeze()
print(y_test_hat_voted)
print(accuracy_score(y_test, y_test_hat_voted))

for i in range(n_estimators):
    print(accuracy_score(y_test, y_test_hats[:,i]))
    
    
# **Soft voting** : Predict class label using the averaging probabilities provided by each individual classifier.
# 확률 사용

y_test_probs = []
for i in range(n_estimators):
    y_test_prob = random_trees[i].predict_proba(X_test)
    y_test_probs.append(y_test_prob)
    
y_test_probs = np.stack(y_test_probs)

print(y_test_probs)

y_test_probs_mean = y_test_probs.mean(axis = 0)
y_test_hat = y_test_probs_mean.argmax(axis = 1)

print(accuracy_score(y_test, y_test_hat))