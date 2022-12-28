import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print(df[df['Sex']!='male'].head(2))

print(df[df['Name']!='Allison, Miss Helen Loraine'].head(2))

print(df[df.index!=0].head(2))