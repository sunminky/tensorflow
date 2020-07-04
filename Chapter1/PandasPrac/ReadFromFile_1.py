import pandas as pd

data_frame = pd.read_csv('friend_list.csv') #파일로부터 데이터 읽기
data_frame = pd.read_csv('friend_list_no_header.txt', #txt파일로 부터 읽기
                         delimiter=',', #구분자는 ","
                         header=None,   #헤더가 없음을 명시
                         names=["name","age","job"]) #헤더 정보 넣어줌
#data_frame.columns = ["name","age","job"]   #헤더 정보 넣어줌

print(data_frame.head(3))   #앞에서 3개의 데이터를 보여줌
print()
print(data_frame.tail(3))   #뒤에서 3개의 데이터를 보여줌
print()
print("name 타입 :",type(data_frame.name))    #각 칼럼들은 시리즈 타입
print("age 타입 :",type(data_frame.age))      #각 칼럼들은 시리즈 타입
print("job 타입 :",type(data_frame["job"]))      #각 칼럼들은 시리즈 타입
print()

s1 = pd.core.series.Series([1,2,3]) #시리즈 만들기, 매개변수로 리스트가 들어감
s2 = pd.core.series.Series(['one','two','three']) #시리즈 만들기, 매개변수로 리스트가 들어감
data_frame2 = pd.DataFrame(data=dict(num=s1, word=s2))    #시리즈들 가지고 데이터 프레임 만들기
print(data_frame2)