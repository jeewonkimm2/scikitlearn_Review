import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print(len(df))

# drop_duplicates() : 아예 똑같아야지만 삭제가 됨.
# len을 확인했을때 결과가 같은 걸 보니 아무것도 삭제 안됨
df = df.drop_duplicates()
print(len(df))

# Sex 열이 모든 같은 행을 중복이라 판단하여 2개(각 female, male)만 남음
df = df.drop_duplicates(subset = ['Sex'])
print(len(df))
print(df)

df = pd.read_csv(url)

# keep : 처음이 아니라 마지막 값들 남기기
df = df.drop_duplicates(subset = ['Sex'], keep = 'last')
print(len(df))
print(df)