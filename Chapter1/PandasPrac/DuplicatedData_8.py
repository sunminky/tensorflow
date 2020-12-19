import pandas as pd

student_list = [{'name': 'John', 'major': "Computer Science", 'sex': "male"},   #중복된 값
                {'name': 'Nate', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Abraham', 'major': "Physics", 'sex': "male"},
                {'name': 'Brian', 'major': "Psychology", 'sex': "male"},
                {'name': 'Janny', 'major': "Economics", 'sex': "female"},
                {'name': 'Yuna', 'major': "Economics", 'sex': "female"},
                {'name': 'Jeniffer', 'major': "Computer Science", 'sex': "female"},
                {'name': 'Edward', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Zara', 'major': "Psychology", 'sex': "female"},
                {'name': 'Sera', 'major': "Economics", 'sex': "female"},    #이름이 중복
                {'name': 'Sera', 'major': "Psychology", 'sex': "female"},   #이름이 중복
                {'name': 'John', 'major': "Computer Science", 'sex': "male"},   #중복된 값
         ]
df = pd.DataFrame(student_list, columns = ['name', 'major', 'sex'])
print(df.duplicated())  #중복된 인덱스 보여줌
print()
print(df.duplicated(["name"]))  #중복된 이름 인덱스를 보여줌
print()
print(df.drop_duplicates()) #중복된 행 제거
print()
print(df.drop_duplicates(["name"], keep="first"))   #이름이 중복된 행 지움, 첫번째행을 제외한 나머지 중복행 삭제(기본값)
print()
print(df.drop_duplicates(["name"], keep="last"))   #이름이 중복된 행 지움, 마지막행을 제외한 나머지 중복행 삭제