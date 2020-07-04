import pandas as pd

friend_list = [
    [15, "student"],
    [25, "developer"],
    [30, "teacher"]
]
column_name = ["age", "job"]
df = pd.DataFrame(friend_list,columns=column_name,index=["John", "Jenny", "Nate"]) #컬림이름, 인덱스 넣어줌
print(df.drop(["John", "Jenny"]))  #John과 Jenny 인덱스인 행 지움
print()
print(df.drop(df.index[[0,2]]))  #0,2 인덱스인 행 지움
print()
print(df.drop("age", axis=1))   #이름 열 없앰
print()
