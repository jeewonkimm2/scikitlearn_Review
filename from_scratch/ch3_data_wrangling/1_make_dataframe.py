import pandas as pd

dataframe = pd.DataFrame()

dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38,25]
dataframe['Driver'] = [True, False]
print(dataframe)
print(dataframe.shape)

new_person = pd.Series(['Molly Mooney', 40, True], index = ['Name', 'Age', 'Driver'])
dataframe = dataframe.append(new_person, ignore_index=True)
print(dataframe)



import numpy as np

data = [['Jacky Jackson', 38, True], ['Steven Stevenson', 25, False]]
matrix = np.array(data)
df1 = pd.DataFrame(matrix, columns=['Name','Age','Driver'])
df2 = pd.DataFrame(data, columns=['Name','Age','Driver'])

print(df1)
print(df2)


data = {'Name' : ['Jacky Jackson','Steven Stevenson'],
        'Age' : [38, 25],
        'Driver' : [True, False]
        }

df3 = pd.DataFrame(data)
print(df3)