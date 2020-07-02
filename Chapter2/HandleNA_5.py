import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

from Chapter2 import downloadData

if __name__ == '__main__':
    downloadData.fetch_housing_data()
    data = downloadData.load_housing_data()

    data["income_cat"] = np.ceil(data["median_income"] / 1.5)  # 중간수입을 1.5로 나눈값을 올림
    # income_cat의 값이 5보다 작지 않으면 5로 세팅
    data["income_cat"].where(data["income_cat"] < 5, 5.0, True)  # 판다스 시리즈에 조건식 적용, where(조건식, 조건안맞을때 바꿀 값, inplace여부)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5,random_state=42)  # 세트의 개수 1개(K폴드 알고리즘), 나누는 비율 0.5, 시드값 42
    for train_index, test_index in split.split(data, data["income_cat"]):  # c2행의 비율을 고려해서 나눔
        start_train_set = data.loc[train_index]  # 인덱스를 기준으로 행을 읽기, iloc은 행번호를 기준으로 행을 읽음
        start_test_set = data.loc[test_index]

    for set_ in (start_train_set, start_test_set):
        set_.drop("income_cat", axis=1, inplace=True)  # income_cat 열 삭제

    housing = start_train_set.drop("median_income", axis=1)
    housing_labels = start_train_set["median_income"].copy()
    
    #결측값 제거 방법
    housing.dropna(subset=["total_bedrooms"])   #1번 방법, 해당 구역 제거
    housing.drop("total_bedrooms", axis=1)  #2번 방법, 전체 특성값 제거
    #3번 방법
    median = housing["total_bedrooms"].median() #중간값 저장
    housing["total_bedrooms"].fillna(median, inplace=True)  #결측값을 모두 평균값으로 채움

    #Imputer를 사용해 결측값 다루기
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")  #결측값을 중간값으로 대체할 것임
    housing_num = housing.drop("ocean_proximity",axis=1)    #텍스트 형은 제외
    imputer.fit(housing_num)    #중간값 추정

    print(imputer.statistics_)   #각 특성의 중간값을 계산해서 저장
    print(housing_num.median().values)   #각 특성의 중간값을 계산해서 저장, 위와 동일
    
    X = imputer.transform(housing_num)  #결측값을 중간값으로 바꾼 데이터(넘파이) 반환
    housing_tr = pd.DataFrame(X, columns=list(housing_num.columns), index=housing.index.values) #넘파일 배열을 다시 판다스 프레임으로 바꿈

    housing_cat = housing["ocean_proximity"]
    housing_cat_encoded, housing_categories = housing_cat.factorize()   #각 카테고리 정수값으로 매칭,
    
    #정수값으로 매칭된 값을 원핫인코딩
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1)) #원핫인코딩, 입력으로 2차월 배열을 넣어야 함, 출력으로 희소행렬 리턴
    housing_cat_1hot_toArr = housing_cat_1hot.toarray() #희소행렬(사이파이)을 밀집행렬(넘파이)로 반환
    print(housing_cat_1hot_toArr)
    print(housing_categories)