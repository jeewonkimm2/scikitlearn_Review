#%%

# Strengths
# - Decision trees work well when you have a mix of continuous and categorical features.
# - The algorithms are completely invariant to scaling of the data. (no data scaling is needed)
# - Feature selection & reduction is automatic.
# - It is robust to noise.
# - The resulting model can easily be visualized and understood.

# Weaknesses
# - They are prone to overfiting.
#  - the ensemble methods are usually used in place of a single decision tree.(You will learn it later in class)
# - It does not take into account interactions between features.
#  - it can only split horizontally on each axis.
# - Space of possible decision trees is exponentially large. Greedy approaches are often unable to find the optimal tree.

import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

# Decision Tree
mglearn.plots.plot_tree_progressive()

# Decision Tree Classifier
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

# Load Dataset
cancer = load_breast_cancer()

print(cancer.keys())
print(cancer.data.shape)
print(cancer.target)

print(np.unique(cancer.target, return_counts=True))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 42)

# model training
clf = DecisionTreeClassifier(random_state = 0)
clf.fit(X_train, y_train)

# model prediction
y_train_hat = clf.predict(X_train)
y_test_hat = clf.predict(X_test)

# evaluation
from sklearn.metrics import accuracy_score
print("Accuracy on training set :{}".format(accuracy_score(y_train, y_train_hat)))
print("Accuracy on test set :{}".format(accuracy_score(y_test, y_test_hat)))


# Varying the hyperparameters
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

training_accuracy = []
test_accuracy = []

msl_settings = [1,2,5,7,10,20]
for msl in msl_settings:
    clf = DecisionTreeClassifier(min_samples_leaf=msl, random_state = 0)
    clf.fit(X_train, y_train)

    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)

    training_accuracy.append(accuracy_score(y_train, y_train_hat))
    test_accuracy.append(accuracy_score(y_test, y_test_hat))
    
result = pd.DataFrame({"min_samples_leaf":msl_settings, "training accuracy": training_accuracy, "test accuracy": test_accuracy})
# 3 and 4 are the best
print(result)


# Visualizing Decision Trees
clf = DecisionTreeClassifier(min_samples_leaf=10, random_state = 0)
clf.fit(X_train, y_train)

from sklearn import tree
import matplotlib.pyplot as plt
# tree.plot_tree(clf)

# If you want to get more clear tree
from sklearn.tree import export_graphviz
# export_graphviz(clf, out_file="tree.dot", class_names=["malignant", "benign"],feature_names=cancer.feature_names, impurity=False, filled=True)


# Feture importance in tress
print("Feature importance : ")
print(clf.feature_importances_)

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plt.figure(figsize=(8,8))
plot_feature_importances_cancer(clf)