import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

# Sex별로 average(평균)
print(df.groupby('Sex').mean())

print(df.groupby('Survived')['Name'].count())

# 두가지 행을 기준으로 그룹핑
print(df.groupby(['Sex','Survived'])['Age'].mean())