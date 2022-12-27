import pandas as pd

# Data url
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print(df.head(2))
print(df.shape)