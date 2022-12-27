# loc : 인덱스가 문자열일때 사용 When index is label(String)
# iloc : 데이터프레임의 위치를 참조


import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

# First row(행)
print(df.iloc[0])

# Second row(행)
print(df.iloc[1])

# Second, third, fourth row(행)
print(df.iloc[1:4])

# First, second, third row(행)
print(df.iloc[:3])

# Set index as name 인덱스를 'Name'으로 설정
df2 = df.set_index(df['Name'])
print(df2)

# 행 확인
print(df2.loc['Allen, Miss Elisabeth Walton'])

# Allison, Miss Helen Loraine 이전 행에서 Age, Sex 열(Column)만 가져옴
print(df2.loc[:'Allison, Miss Helen Loraine', 'Age':'Sex'])

print(df2[['Age','Sex']].head(2))