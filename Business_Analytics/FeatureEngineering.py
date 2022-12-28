import os
import mglearn
import pandas as pd
import numpy as np

adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(
    adult_path, header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])


print(data.workclass.value_counts())
print(data.occupation.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = data.drop("income", axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, data.income, random_state=0)

# Categorical and Numerical 나누기
X_train_cat = X_train[['workclass','education','gender','occupation']]
X_train_num = X_train[['age','hours-per-week']]

X_test_cat = X_test[['workclass','education','gender','occupation']]
X_test_num = X_test[['age','hours-per-week']]

# For Categorical Values
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse = False)
ohe.fit(X_train_cat)

X_train_cat_ohe = ohe.transform(X_train_cat)
X_test_cat_ohe = ohe.transform(X_test_cat)

# For Numerical Values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_num)

X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Categorical and Numerical 합치기
X_train_trans = np.concatenate([X_train_cat_ohe, X_train_num_scaled], axis=1)
X_test_trans = np.concatenate([X_test_cat_ohe, X_test_num_scaled], axis=1)


logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train_trans, y_train)

y_test_hat = logreg.predict(X_test_trans)
print("Accuracy on testing set: ", accuracy_score(y_test, y_test_hat))


# ColumnTransformer : 앞의 과정을 더 쉽게 해줌
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = data.drop("income", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, data.income, random_state=0)

ct = ColumnTransformer([("scaling", StandardScaler(), ['age', 'hours-per-week']), ("onehot", OneHotEncoder(sparse=False),['workclass', 'education', 'gender', 'occupation'])])

ct.fit(X_train)
X_train_trans = ct.transform(X_train)

logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train_trans, y_train)

X_test_trans = ct.transform(X_test)
y_test_hat = logreg.predict(X_test_trans)

# 같은 결과
print("Accuracy on testing set: ", accuracy_score(y_test, y_test_hat))