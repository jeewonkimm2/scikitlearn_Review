import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

print(df['Sex'].replace("female","Woman").head(2))

print(df['Sex'].replace(["female","male"],["Woman","Man"]).head(5))

# 전체 DataFrame에서 1 -> One
print(df.replace(1,"One").head(2))

# 정규표현식(regular expression)도 인식함
print(df.replace(r"1st","First",regex=True).head(2))

# female, male을 person으로 바꿈
print(df.replace(["female","male"], "person").head(3))

# female : 1, male : 0
print(df.replace({"female":1, "male":0}).head(3))