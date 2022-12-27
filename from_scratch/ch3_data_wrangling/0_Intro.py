# Data Wrangling?
# 원본 데이터를 정제하고 사용 가능한 형태로 구성하기 위한 변환 과정을 광범위하게 의미하는 비공식적인 용어

import pandas as pd

# Data url
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'

# Data loading into dataframe 데이터프레임으로 데이터 적재
dataframe = pd.read_csv(url)

print(dataframe.shape)

# print(dataframe.head(5))