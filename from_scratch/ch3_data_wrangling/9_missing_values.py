import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print(df[df['Age'].isnull()].head(2))


# NaN을 사용하려면 numpy 사용해야함
import numpy as np

df['Sex'] = df['Sex'].replace('male', np.nan)

print(df)