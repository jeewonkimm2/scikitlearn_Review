# concat 함수에 axis=0(Row 방향) axis=1(Column 방향) 매개변수를 설정하여 행의 축을 띠라 연결하기

import pandas as pd

data_a = {'id':['1','2','3'],
          'first':['Alex','Amy','Allen'],
          'last':['Anderson','Ackerman','Ali']}

dataframe_a = pd.DataFrame(data_a, columns = ['id','first','last'])

data_b = {'id':['4','5','6'],
          'first':['Billy','Brian','Bran'],
          'last':['Bonder','Black','Balwner']}

dataframe_b = pd.DataFrame(data_b, columns = ['id','first','last'])


dataframe_row = pd.concat([dataframe_a, dataframe_b], axis = 0)
print(dataframe_row)

dataframe_column = pd.concat([dataframe_a, dataframe_b], axis = 1)
print(dataframe_column)