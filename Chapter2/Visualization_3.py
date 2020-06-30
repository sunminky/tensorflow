from Chapter2 import downloadData
import matplotlib.pyplot as plt

if __name__ == '__main__':
    downloadData.fetch_housing_data()
    data = downloadData.load_housing_data()

    #모든 구역을 산점도로 만들어서 시각화
    #data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)  #투명도 0.1
    #plt.show()

    # 모든 구역을 산점도로 만들어서 시각화
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, # 투명도 0.4,
              s=data["population"]/100,  #원의 크기(s)는 인구에 비례
              label="population",       #population이라고 라벨
              figsize=(10,7),           #10 X 7 의 크기로 그래프 표시
              c="median_house_value",   #색깔은 중간 주택 가격에 영향
              cmap=plt.get_cmap("jet"), #컬러맵은 파란색~빨간색(jet) 사용
              colorbar=True,           #옆에 나타나는 칼러바 사용
              sharex=False  #모든 서브플롯이 같은 x축 눈금을 사용하지 않음, 여기서는 없어도 됌
              )
    plt.show()