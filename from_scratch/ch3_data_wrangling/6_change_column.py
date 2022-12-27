import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

# Column이름 바꾸기 PClass -> Passenger Class
print(df.rename(columns={"PClass":"Passenger Class"}).head(2))

# 두 개의 Column이름 바꾸기
print(df.rename(columns={"PClass":"Passenger Class", "Sex":"Gender"}).head(2))



import collections

column_names = collections.defaultdict(str)

for name in df.columns:
    column_names[name]
    
print(column_names)