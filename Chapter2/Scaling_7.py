from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np

if __name__ == '__main__':
    data = np.linspace(5,10,20).reshape(-1,1)

    #데이터가 평균 중심으로 정규분포를 이루도록 표준화
    stdScaler = StandardScaler()
    stdScaler.fit(data)
    print(stdScaler.transform(data))

    print("-----------------------------------")

    #데이터를 0~1사이에 들어오도록 정규화
    mmScaler = MinMaxScaler()
    mmScaler.fit(data)
    print(mmScaler.transform(data))