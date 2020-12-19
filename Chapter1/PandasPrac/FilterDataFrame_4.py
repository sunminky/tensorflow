import pandas as pd

friend_list = [
    ["John", 25, "student"],
    ["Nate", 30, "teacher"],
    ["Jenny", 30, None]
]
column_name = ["name", "age", "job"]
df = pd.DataFrame(friend_list,columns=column_name)

#조건에 맞는 행 보여주기
print(df[1:3]) #슬라이싱 가능
print()
print(df.loc[[0,2]])  #0행과 2행 보여줌
print()
print(df.iloc[[0,2]])  #0, 2번 인덱스 보여줌
print()
print(df[(df.age>25)&(df.name=="Jenny")])    #조건에 맞는 행만 출력
print()
print(df.query("age>25 & name==\"Nate\""))   #조건에 맞는 행만 출력
print()

#조건에 맞는 열 보여주기
print(df.iloc[:,0:2])   #0~1번열의 모든 행 표시
print()
print(df[["name","job"]])   #이름과 직업만 표시
print()
print(df.filter(items=["name","job"]))  #이름과 직업만 표시
print()
print(df.filter(like="a", axis=1))  #이름에 a가 들어간 컬럼표시
print()
print(df.filter(regex="b$", axis=1))    #이름이 b로 끝나는 컬럼표시
print()