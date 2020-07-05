import pandas as pd

l1 = [{'name': 'John', 'job': "teacher"},
      {'name': 'Nate', 'job': "student"},
      {'name': 'Fred', 'job': "developer"}]

l2 = [{'name': 'Ed', 'job': "dentist"},
      {'name': 'Jack', 'job': "farmer"},
      {'name': 'Ted', 'job': "designer"}]

df1 = pd.DataFrame(l1, columns=['name', 'job'])
df2 = pd.DataFrame(l2, columns=['name', 'job'])

result_row = pd.concat([df1,df2], ignore_index=True)   #행 기준으로 합침, 기존 인덱스 무시
print(result_row)
print()

result_row2 = df1.append(df2, ignore_index=True)   #df1 뒤에 행 기준으로 합침, 기존 인덱스 무시
print(result_row2)
print()

l3 = [{'name': 'John', 'job': "teacher"},
      {'name': 'Nate', 'job': "student"},
      {'name': 'Jack', 'job': "developer"}]

l4 = [{'age': 25, 'country': "U.S"},
      {'age': 30, 'country': "U.K"},
      {'age': 45, 'country': "Korea"}]

df1 = pd.DataFrame(l3, columns=['name', 'job'])
df2 = pd.DataFrame(l4, columns=['age', 'country'])

result_column = pd.concat([df1,df2], axis=1, ignore_index=False)   #열 기준으로 합침, 기존 인덱스(컬럼이름) 무시안함
print(result_column)
print()