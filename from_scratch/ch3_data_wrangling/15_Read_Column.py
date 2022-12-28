import pandas as pd

url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

df = pd.read_csv(url)

# 처음 두개를 대문자로 바꿔서 출력
for name in df['Name'][0:2]:
    print(name.upper())