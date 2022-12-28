# **Strengths**
# - Linear models are very fast to train, and also fast to predict.
# - They scale to very large datasets and work well with sparse data.
# - They make it relatively easy to understand how a prediction is made.

# **Weaknesses**
# - If your dataset has highly correlated features, it is often not entirely clear why coefficients are the way they are. (It is important to remove redundant features – feature selection)
# - They would perform worse if the relationship between features and target in your dataset is non-linear.

import mglearn
X, y = mglearn.datasets.load_extended_boston()
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
y_train_hat = lr.predict(X_train)
y_test_hat = lr.predict(X_test)

print('Performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('Performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))


# Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 1)
ridge.fit(X_train, y_train)

y_train_hat = ridge.predict(X_train)
y_test_hat = ridge.predict(X_test)

print('Ridge Performance for TRAIN--------')
print('train MSE : ', mean_squared_error(y_train, y_train_hat))
print('Ridge Performance for TEST--------')
print('test MSE : ', mean_squared_error(y_test, y_test_hat))


# Varying the hyperparameter
training_mse = []
test_mse = []
alpha_settings = [0,0.1,1,10]

for alpha in alpha_settings:
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train, y_train)
    y_train_hat = ridge.predict(X_train)
    y_test_hat = ridge.predict(X_test)
    training_mse.append(mean_squared_error(y_train,y_train_hat))
    test_mse.append(mean_squared_error(y_test,y_test_hat))
    
import pandas as pd
# when alpha = 0.1 is best
print(pd.DataFrame({"alpha":alpha_settings, "training MSE": training_mse, "test MSE": test_mse}))


import matplotlib.pyplot as plt

# When α=10, the coefficients are mostly between around –3 and 3
# When α=0 (Linear Regression), the coefficients have larger magnitude

ridge = Ridge(alpha=1).fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()


# Lasso

from sklearn.linear_model import Lasso

training_mse = []
test_mse = []
num_vars_used = []

alpha_settings = [0.0001, 0.001, 0.01, 0.1, 1]
for alpha in alpha_settings:
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train)
    num_vars_used.append(sum(lasso.coef_!=0))
    
    y_train_hat = lasso.predict(X_train)
    y_test_hat = lasso.predict(X_test)

    training_mse.append(mean_squared_error(y_train, y_train_hat))
    test_mse.append(mean_squared_error(y_test, y_test_hat))
    
# when alpha = 0.01 is the best
print(pd.DataFrame({"alpha":alpha_settings, "training MSE": training_mse, "test MSE": test_mse, "variables used": num_vars_used}))

lasso = Lasso().fit(X_train, y_train)
lasso001 = Lasso(alpha=0.01, max_iter=1000).fit(X_train, y_train)
lasso00001 = Lasso(alpha=0.0001, max_iter=1000).fit(X_train, y_train)

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")


# Logistic Regression
# penalty : default L2 (similar to ridge)
# penalty 파라미터를 l1으로 바꾸면 Lisso와 비슷함
# C (inverse of alpha)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

training_accuracy = []
test_accuracy = []
C_settings = [0.01, 0.1, 1, 10, 100, 1000, 10000]

for C in C_settings:
    lr = LogisticRegression(C = C)
    lr.fit(X_train, y_train)
    y_train_hat = lr.predict(X_train)
    y_test_hat = lr.predict(X_test)
    training_accuracy.append(accuracy_score(y_train, y_train_hat))
    test_accuracy.append(accuracy_score(y_test,y_test_hat))
    
print(pd.DataFrame({"C":C_settings, "training accuracy": training_accuracy, "test accuracy": test_accuracy}))