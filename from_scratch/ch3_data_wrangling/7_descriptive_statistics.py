# Descriptive Statistics 기술통계

import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print("MAX : {}".format(df["Age"].max()))
print("MIN : {}".format(df["Age"].min()))
print("AVG : {}".format(df["Age"].mean()))
print("SUM : {}".format(df["Age"].sum()))
print("COUNT : {}".format(df["Age"].count()))
print("SKEW : {}".format(df["Age"].skew()))

print(df.count())

print("COVARIANCE 공분산 : {}".format(df.cov()))

print("CORRELATION 상관계수 : {}".format(df.corr()))