import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

# 대문자로 바꾸는 함수
def uppercase(x):
    return x.upper()

# apply : 함수 적용
# 함수 적용 후 출력
print(df['Name'].apply(uppercase))

# map : apply와 흡사하나, Dictionary를 입력값으로 넣을 수 있음
print(df['Survived'].map({1:'Live',2:'Dead'})[:5])

# 30살보다 어린 나이를 가졌는지
print(df['Age'].apply(lambda x, age: x<age, age=30)[:5])

# 가장 큰 값
print(df.apply(lambda x :max(x)))

# applymap : map과 비슷하며 열의 각 원소에 적용
def truncate_string(x):
    if type(x) == str:
        return x[:20]
    return x

print(df.applymap(truncate_string)[:5])