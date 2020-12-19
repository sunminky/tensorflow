import pandas as pd

friend_dict_list = [
    {"name": "John", "age": 25, "job": "student"},
    {"name": "Nate", "age": 30, "job": "teacher"}
]

df = pd.DataFrame(friend_dict_list) #데이터프레임 생성
df = df[["name","age","job"]] # 컬럼이 입력한 순서대로 정렬이 안되서 재정렬 필요
print(df.head())
print()

###입력 순서가 보장되게 하자###
from collections import OrderedDict

friend_ordered_dict = OrderedDict(
    [
        ("name",["John","Nate"]),
        ("age",[25,30]),
        ("job",["student", "teacher"])
    ]
)

df2 = pd.DataFrame(friend_ordered_dict) #데이터프레임 생성, 입력순서 보장
print(df2.head())
print()

friend_list = [
    ["John", 25, "student"],
    ["Nate", 30, "teacher"]
]
column_name = ["name", "age", "job"]
df3 = pd.DataFrame(friend_list,columns=column_name,index=list(range(len(friend_list)))) #컬림이름, 인덱스 넣어줌
print(df3.head())