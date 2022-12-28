import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print(df.groupby('Sex').apply(lambda x:x.count()))