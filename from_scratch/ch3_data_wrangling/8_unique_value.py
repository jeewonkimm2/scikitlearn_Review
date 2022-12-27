import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print(df['Sex'].unique())

print(df['Sex'].value_counts())

print(df['PClass'].value_counts())

# 고유한 개수
print(df['PClass'].nunique())

# 전체 DataFrame
# dropna = True : NaN 카운트 X, 기본값
print(df.nunique(dropna=True))