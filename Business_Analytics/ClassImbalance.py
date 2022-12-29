#%%

import numpy as np
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

digits = load_digits()

X = digits.data
y = digits.target == 9

np.unique(y, return_counts = True)

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, stratify=y, random_state=0) 
# stratified split could be important because this is an imbalanced dataset.

tree = DecisionTreeClassifier(max_depth=20).fit(X_train, y_train) 
y_test_hat = tree.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# false negative 는 두번째 row 첫번째 컬럼
print(confusion_matrix(y_test, y_test_hat))
print("Test score: {:.2f}".format(accuracy_score(y_test, y_test_hat)))
print("Test f1 score: {:.2f}".format(f1_score(y_test, y_test_hat)))

# this is how to draw the ROC curve
# _ 는 threshold이다
fpr, tpr, _ = roc_curve(y_test, tree.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
print("AUC : {}".format(roc_auc_score(y_test , tree.predict_proba(X_test)[:,1]))) # predicted probabilities of postive


# Cost sensitive

# All we have to do is set the 'class_weight'.
# Almost all models in scikit-learn provides the 'class_weight' argument that can be specified as a model hyperparameter.
tree = DecisionTreeClassifier(max_depth = 20, class_weight = 'balanced').fit(X_train, y_train)
y_test_hat = tree.predict(X_test)

# class_weight='balanced'를 쓰면
# 1212:negative, 135:positive => weight가 proportional하게 바뀐다 1/1212:1/135

print(np.unique(y_train, return_counts=True))

print("Test score: {:.2f}".format(accuracy_score(y_test, y_test_hat)))
print("Test f1 score: {:.2f}".format(f1_score(y_test, y_test_hat)))
print(confusion_matrix(y_test, y_test_hat))

fpr, tpr, _ = roc_curve(y_test , tree.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
print("AUC : {}".format(roc_auc_score(y_test , tree.predict_proba(X_test)[:,1])))


# Random oversampling
over_no = y_train.sum()*9

positive_index=y_train.nonzero()[0]
over_index = np.random.choice(positive_index,over_no, replace=True)

X_train_add = X_train[over_index,:]
y_train_add = y_train[over_index]

X_train_over = np.concatenate((X_train,X_train_add))
y_train_over = np.concatenate((y_train,y_train_add))

np.unique(y_train_over, return_counts = True)

tree = DecisionTreeClassifier(max_depth = 20).fit(X_train_over,y_train_over)
y_test_hat = tree.predict(X_test)

print("Test score: {:.2f}".format(accuracy_score(y_test, y_test_hat)))
print("Test f1 score: {:.2f}".format(f1_score(y_test, y_test_hat)))
print(confusion_matrix(y_test, y_test_hat))

fpr, tpr, _ = roc_curve(y_test , tree.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
print("AUC : {}".format(roc_auc_score(y_test , tree.predict_proba(X_test)[:,1])))
# oversampling does not necessarily lead to better results.


# Random undersampling

X_train_major = X_train[y_train==0]
X_train_minor = X_train[y_train==1]

y_train_major = y_train[y_train==0]
y_train_minor = y_train[y_train==1]

# replace=Fasle는 디자인의 차이임!
# 위에 나와있음
under_no = 135

under_idx = np.random.choice(X_train_major.shape[0],under_no,replace=False)

X_train_major_under = X_train_major[under_idx,:]
y_train_major_under = y_train_major[under_idx]

X_train_under = np.concatenate((X_train_minor, X_train_major_under))
y_train_under = np.concatenate((y_train_minor, y_train_major_under))

np.unique(y_train_under, return_counts=True)

tree = DecisionTreeClassifier(max_depth=20).fit(X_train_under,y_train_under)
y_test_hat = tree.predict(X_test)

print("Test score: {:.2f}".format(accuracy_score(y_test, y_test_hat)))
print("Test f1 score: {:.2f}".format(f1_score(y_test, y_test_hat)))
print(confusion_matrix(y_test, y_test_hat))

fpr, tpr, _ = roc_curve(y_test , tree.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
print("AUC : {}".format(roc_auc_score(y_test , tree.predict_proba(X_test)[:,1])))



### try to use imblearn package
# - https://imbalanced-learn.org/stable/index.html