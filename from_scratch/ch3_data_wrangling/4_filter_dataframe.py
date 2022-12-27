import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

# Print Sex == female 2개
print(df[df['Sex']=='female'].head(2))

# 65세 이상 female
print(df[(df['Sex']=='female')&(df['Age']>=65)])