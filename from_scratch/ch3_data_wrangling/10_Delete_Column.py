import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print(df.drop('Age', axis=1).head(2))

print(df.drop(['Age','Sex'], axis = 1).head(2))

print(df.drop(df.columns[1], axis=1).head(2))

# drop후 새로운 데이터 만들기
df_new = df.drop(df.columns[0], axis = 1)

print(df_new)