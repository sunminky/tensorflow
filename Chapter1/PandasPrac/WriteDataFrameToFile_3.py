import pandas as pd

friend_list = [
    ["John", 25, "student"],
    ["Nate", 30, "teacher"],
    ["Jenny", 30, None]
]
column_name = ["name", "age", "job"]
df = pd.DataFrame(friend_list,columns=column_name)
df.to_csv("save_df.csv",    #데이터프레임을 파일로 저장
          index=True,       #인덱스 저장
          header=True,       #헤더 저장
          na_rep="-")       #결측값은 -로 치환