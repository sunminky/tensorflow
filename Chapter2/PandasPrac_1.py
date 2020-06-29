from Chapter2 import downloadData
import pandas as pd
import matplotlib.pyplot as plt
from Chapter2 import generateData_2

# 최대 줄 수 설정
pd.set_option('display.max_rows', 500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 500)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    downloadData.fetch_housing_data()
    data = downloadData.load_housing_data()
    print(data.head(10))    #위에서부터 10개의 데이터 보여줌
    print(data.tail(10))    #아래에서부터 10개의 데이터 보여줌
    print(data.info())      #데이터에 대한 간단한 설명, 전체 행 수, 특성의 타입과 널이 아닌 개수 표시
    print(data["ocean_proximity"].value_counts())   #각 카테고리 별 개수 표시
    print(data.describe())  #숫자형 특성의 요약정보 표시
    data.hist(bins=50, figsize=(20,15)) #구간의 개수 50개, 가로 20 세로 15의 크기로 표시
    plt.show()