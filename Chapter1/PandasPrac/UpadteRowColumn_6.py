import pandas as pd
import numpy as np

friend_list = [
    ["John", 15, "student"],
    ["Jenny", 30, "developer"],
    ["Nate", 30, "teacher"]
]
column_name = ["Name", "age", "job"]
df = pd.DataFrame(friend_list,columns=column_name)

df["salary"] = 0    #열 추가하기
print(df)
print()

df["salary"] = np.where(df["job"] != "student", 'yes', 'no')    #조건식 만족하면 yes, 만족안하면 no
print(df)
print()

friend_exam_list = [
    ["John", 95, 85],
    ["Jenny", 85, 80],
    ["Nate", 30, 10]
]
column_name = ["Name", "midterm", "final"]
df2 = pd.DataFrame(friend_exam_list,columns=column_name)
df2["total"] = df2["midterm"] + df2["final"]    #열을 더함
print(df2)
print()

df2["grade"] = ["A","B","F"]    #새로운 열에 값 대입
print(df2)
print()

def pass_or_fail(row):
    if row != 'F':
        return "Pass"
    else:
        return "Fail"

df2.grade = df2.grade.apply(pass_or_fail) #열의 값에 함수를 적용시킴
print(df2)
print()

new_data = [
    ["Ben",50,50,100,"Pass"]
]
new_column_name = ["Name", "midterm", "final", "total", "grade"]
df3 = pd.DataFrame(new_data,columns=new_column_name)

print(df2.append(df3, ignore_index=True))  #데이터 프레임 합치기, 합치면서 인덱스 무시

