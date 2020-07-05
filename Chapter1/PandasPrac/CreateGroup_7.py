import pandas as pd

student_list = [{'name': 'John', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Nate', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Abraham', 'major': "Physics", 'sex': "male"},
                {'name': 'Brian', 'major': "Psychology", 'sex': "male"},
                {'name': 'Janny', 'major': "Economics", 'sex': "female"},
                {'name': 'Yuna', 'major': "Economics", 'sex': "female"},
                {'name': 'Jeniffer', 'major': "Computer Science", 'sex': "female"},
                {'name': 'Edward', 'major': "Computer Science", 'sex': "male"},
                {'name': 'Zara', 'major': "Psychology", 'sex': "female"},
                {'name': 'Wendy', 'major': "Economics", 'sex': "female"},
                {'name': 'Sera', 'major': "Psychology", 'sex': "female"}
         ]
df = pd.DataFrame(student_list, columns = ['name', 'major', 'sex'])
groupby_major = df.groupby('major') #전공별로 그룹지음
print(groupby_major.groups) #그룹 표시
print()
for name, group in groupby_major:   #그룹 표시
    print(name + " : " + str(len(group)))
    print(group)
    print()

df_major_cnt = pd.DataFrame({"count":groupby_major.size()}).reset_index()   #major(그룹을 나누는 기준이 되는 열)이 인덱스가 되지 않도록 reset_index로 인덱스 추가
print(df_major_cnt)