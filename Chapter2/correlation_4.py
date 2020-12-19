from Chapter2 import downloadData
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

if __name__ == '__main__':
    downloadData.fetch_housing_data()
    data = downloadData.load_housing_data()

    corr_matrix = data.corr()   #특성간의 상관관계 표시
    print(corr_matrix)  #1에 가까우면 강한 양의 상관관계, -1에 가까우면 강한 음의 상관관계

    print(corr_matrix["median_income"].sort_values(ascending=False))    #median_income과 상관관계를 내림차순으로 정렬

    ### 그래프를 사용해서 상관관계 분석하기 ###
    attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
    scatter_matrix(data[attributes], figsize=(12,8))
    plt.show()  #4개의 특성의 조합, 총16개의 그래프가 나옴

    ### 하나씩 자세히 보기 ###
    data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()  #상관관계가 너무 강함(포인트들이 널리 퍼지지 않음), 최대 상한선이 500000으로 고정되어 있음