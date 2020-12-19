import pandas as pd

school_id_list = [{'name': 'John', 'job': "teacher", 'age': 40},
                {'name': 'Nate', 'job': "teacher", 'age': 35},
                {'name': 'Yuna', 'job': "teacher", 'age': 37},
                {'name': 'Abraham', 'job': "student", 'age': 10},
                {'name': 'Brian', 'job': "student", 'age': 12},
                {'name': 'Janny', 'job': "student", 'age': 11},
                {'name': 'Nate', 'job': "teacher", 'age': None},
                {'name': 'John', 'job': "student", 'age': None}
         ]
df = pd.DataFrame(school_id_list, columns = ['name', 'job', 'age'])
print(df.info())    #데이터프레임 정보 보여줌, NA값 확인가능
print()
print(df.isna())    #NA값 보여줌
print()
print(df.isnull())    #NA값 보여줌
print()

dfcopy = df.copy()
dfcopy.age = dfcopy.age.fillna(0)   #NA값을 0으로 바꿈
print(dfcopy)
print()

dfcopy = df.copy()
dfcopy["age"].fillna(df.groupby("job")["age"].transform("median"), inplace=True)    #직업별로 그룹화, 그룹별 평균 나이로 NA값 치환
print(dfcopy)