# Grid search with cross-validation

# We will use Lasso regression model.
# We will use R2 as an evaluation metric (Possible for regression model)
# Conduct grid search on the hyperparameter alpha with 5-fold cross-validation. 
# Finally, check the final generalization performance on the test set with the chosen hyperparameter.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

diabetes = datasets.load_diabetes()

# print(diabetes.data)

X_trainval, X_test, y_trainval, y_test = train_test_split(diabetes.data, diabetes.target, test_size = 0.2, random_state=0)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
scaler = StandardScaler()

best_score = 0

# logspace : a, b, c 을 log scale 등간격인 행 벡터를 생성하는  명령어
# 예시
# log10(x1)=2, log10(x2)=3, log10(x3)=4 만들기
# np.logspace(2, 4, 3)
# 결과
# array([  100.,  1000., 10000.])
for alpha in np.logspace(-4,1,30):
    scores_val = []
    for train_idx, val_idx in kfold.split(X_trainval, y_trainval):
        X_train = X_trainval[train_idx]
        y_train = y_trainval[train_idx]
        X_valid = X_trainval[val_idx]
        y_valid = y_trainval[val_idx]

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        lasso = Lasso(alpha = alpha, random_state = 0, max_iter = 10000)
        lasso.fit(X_train_scaled, y_train)

        y_valid_hat = lasso.predict(X_valid_scaled)
        scores_val.append(r2_score(y_valid, y_valid_hat))
        
    mean_score = np.mean(scores_val)

    if mean_score > best_score:
        best_score = mean_score
        best_parameters = {'alpha':alpha}
    
    
print("Best score on validation set: {:.7f}".format(best_score))
print("Best hyperparameters: {}".format(best_parameters))

scaler.fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso(**best_parameters, random_state = 0, max_iter = 10000)
lasso.fit(X_trainval_scaled, y_trainval)

y_test_hat = lasso.predict(X_test_scaled)
test_score = r2_score(y_test, y_test_hat)
print("Test set score with best hyperparameters: {:.7f}".format(test_score))



# Evaluatio Metric
# Get the probability of belonging to each class
# Draw confusion matrix and calcuate the recall and precision.
# Change the threshold and check the change of the value of metric.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_val, y_train, y_val = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

clf = KNeighborsClassifier(n_neighbors = 30)
clf.fit(X_train_scaled, y_train)

y_val_hat = clf.predict(X_val_scaled)
# 각 클래스에 대한 확률
y_val_prob = clf.predict_proba(X_val_scaled)

prob_positive = y_val_prob[:,1]

print(prob_positive > 0.5)

# Threshold 조정
y_val_hat = prob_positive > 0.3
print(y_val_hat)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_val, y_val_hat)
print("Confusion matrix:\n{}".format(confusion))

recall = confusion[1,1]/confusion[1,:].sum()
precision = confusion[1,1]/confusion[:,1].sum()

print('recall : '.format(recall))
print('precision : '.format(precision))