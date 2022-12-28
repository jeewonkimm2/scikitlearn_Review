import pandas as pd
import numpy as np

time_index = pd.date_range('06/06/2017', periods = 100000, freq = '30S')
print(time_index)

# Dataframe 인덱스를 time_index로
df = pd.DataFrame(index = time_index)

# random values로 컬럼 만들기
df['Sale_Amount'] = np.random.randint(1,10,100000)
print(df)


# 일(Day)단위로 시간간격 재조정
print(df.resample('D').sum())

# 주(Week)단위로 시간간격 재조정
print(df.resample('W').sum())

# 2주(Week)단위로 시간간격 재조정
print(df.resample('2W').sum())

# 월(Month)단위로 시간간격 재조정
print(df.resample('M').sum())

# 월(Month) 시작 날짜 단위로 시간간격 재조정
print(df.resample('MS').sum())


# 년(Year)단위로 시간간격 재조정
print(df.resample('Y').sum())