import pandas as pd
import numpy as np

def extract_year(column):
    return column.split("-")[0]

def get_age(year, current_year):
    return current_year - int(year)

def yearNage(row):  #행을 입력받음
    return "current year is " + str(row.year) + " my age is " + str(row.age) #행의 원소들 사용

date_list = [{'yyyy-mm-dd': '2000-06-27'},
         {'yyyy-mm-dd': '2002-09-24'},
         {'yyyy-mm-dd': '2005-12-20'}]
df = pd.DataFrame(date_list, columns = ['yyyy-mm-dd'])
###방법1(apply 함수 사용)###
df["year"] = df['yyyy-mm-dd'].apply(extract_year)   #year 컬럼에 extract_year 함수 적용
print(df)
print()
df["age"] = df["year"].apply(get_age, current_year=2020)    #apply로 적용하는 함수에 매개변수 전달
print(df)
print()
df["introduce"] = df.apply(yearNage, axis=1)    #매개변수로 행을 전달, axis=1은 열, axis=0은 행
print(df)
print()

###방법2(map, applymap 함수 사용)###
df = pd.DataFrame(date_list, columns = ['yyyy-mm-dd'])
df["year"] = df['yyyy-mm-dd'].map(extract_year) #year 컬럼에 extract_year 함수 적용
print(df)
print()

job_list = [{'age': 20, 'job': 'student'},
         {'age': 30, 'job': 'developer'},
         {'age': 30, 'job': 'teacher'}]
df2 = pd.DataFrame(job_list)

df2.job = df2.job.map({"student":1, "developer":2, "teacher":3})  #문자열 데이터를 숫자형으로 바꾸기
print(df2)
print()

x_y = [{'x': 5.5, 'y': -5.6, 'z':-1.1},
         {'x': -5.2, 'y': 5.5, 'z':-2.2},
         {'x': -1.6, 'y': -4.5, 'z':-3.3}]
df3 = pd.DataFrame(x_y)

df3 = df3.applymap(np.around) #모든 행과 열에 반올림 함수 적용하기
print(df3)
print()