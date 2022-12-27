#%%

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

# iris data import
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 85)

# gini impurity
t1 = DecisionTreeClassifier(max_depth = None, min_samples_split = 2, min_samples_leaf = 1, criterion = 'gini')
# entropy impurity
t1 = DecisionTreeClassifier(max_depth = None, min_samples_split = 2, min_samples_leaf = 1, criterion = 'entropy')

t1.fit(X_train, y_train)
y_pred = t1.predict(X_train)
y_prob = t1.predict_proba(X_train)

print(y_prob)

print(t1.score(X_train, y_train))
print(t1.score(X_test, y_test))


# breast data import
breast = datasets.load_breast_cancer()

X2 = breast.data
y2 = breast.target

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.2, shuffle = True, random_state = 85)

t2 = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 10)
t2.fit(X_train2, y_train2)
print(t2.score(X_train2, y_train2))
print(t2.score(X_test2, y_test2))
y_prob2 = t2.predict_proba(X_train2)
print(y_prob2)


from sklearn import tree
import matplotlib.pyplot as plt

tree.plot_tree(t2)

fig = plt.figure(dpi=400)
tree.plot_tree(t2, feature_names=breast.feature_names, class_names = breast.target_names, filled = True)


# boston data import
boston = datasets.load_boston()

X3 = boston.data
y3 = boston.target

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size = 0.2, shuffle = True, random_state = 85)

from sklearn.metrics import mean_squared_error

t3 = DecisionTreeRegressor(max_depth = 5, criterion = 'friedman_mse')
t3.fit(X_train3, y_train3)
y_pred_train = t3.predict(X_train3)
y_pred_test = t3.predict(X_test3)

print(mean_squared_error(y_train3, y_pred_train))
print(mean_squared_error(y_test3, y_pred_test))