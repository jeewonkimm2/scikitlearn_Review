# 내부 병합 Inner Join을 하려면 on 매개변수에 병합 열을 지정하여 merge 메서드를 하자

import pandas as pd

employee_data = {'employee_id':['1','2','3','4'],
                 'name':['Amy Jones', 'Allen Keys',' Alice Bees','Tim Horton']}

df_employees = pd.DataFrame(employee_data, columns = ['employee_id','name'])
print(df_employees)


sales_data = {'employee_id':['3','4','5','6'],
                 'total_sales':[23456,2512,2345,1455]}

df_sales = pd.DataFrame(sales_data, columns = ['employee_id','total_sales'])


# Inner Join(Default 값)
df_IJ = pd.merge(df_employees, df_sales, on = 'employee_id')
print(df_IJ)

# Outer Join
df_OJ = pd.merge(df_employees, df_sales, on = 'employee_id', how = 'outer')
print(df_OJ)

# Left Join
df_LJ = pd.merge(df_employees, df_sales, on = 'employee_id', how = 'left')
print(df_LJ)

# Right Join
df_RJ = pd.merge(df_employees, df_sales, on = 'employee_id', how = 'right')
print(df_RJ)