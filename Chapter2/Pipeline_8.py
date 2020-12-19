import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Chapter2 import MyTransform_6, downloadData

if __name__ == '__main__':
    downloadData.fetch_housing_data()
    data = downloadData.load_housing_data()

    data["income_cat"] = np.ceil(data["median_income"] / 1.5)  # 중간수입을 1.5로 나눈값을 올림
    # income_cat의 값이 5보다 작지 않으면 5로 세팅
    data["income_cat"].where(data["income_cat"] < 5, 5.0, True)  # 판다스 시리즈에 조건식 적용, where(조건식, 조건안맞을때 바꿀 값, inplace여부)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)  # 세트의 개수 1개(K폴드 알고리즘), 나누는 비율 0.5, 시드값 42
    for train_index, test_index in split.split(data, data["income_cat"]):  # c2행의 비율을 고려해서 나눔
        start_train_set = data.loc[train_index]  # 인덱스를 기준으로 행을 읽기, iloc은 행번호를 기준으로 행을 읽음
        start_test_set = data.loc[test_index]

    for set_ in (start_train_set, start_test_set):
        set_.drop("income_cat", axis=1, inplace=True)  # income_cat 열 삭제

    housing = start_train_set.drop("median_income", axis=1)
    housing_labels = start_train_set["median_income"].copy()

    # 결측값 제거 방법
    housing.dropna(subset=["total_bedrooms"])  # 1번 방법, 해당 구역 제거
    housing.drop("total_bedrooms", axis=1)  # 2번 방법, 전체 특성값 제거
    # 3번 방법
    median = housing["total_bedrooms"].median()  # 중간값 저장
    housing["total_bedrooms"].fillna(median, inplace=True)  # 결측값을 모두 평균값으로 채움

    # Imputer를 사용해 결측값 다루기
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")  # 결측값을 중간값으로 대체할 것임
    housing_num = housing.drop("ocean_proximity", axis=1)  # 텍스트 형은 제외

    ########파이프 라인###################
    num_attribs = housing_num.columns
    cat_attribs = "ocean_proximity"

    num_pipline = Pipeline([
        ('selector',MyTransform_6.DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', MyTransform_6.CombinedAttributesAdder()),
        ('stdScaler',StandardScaler()),
    ])  #Pipeline은 연속된 단계를 나타내는 이름/추정기 쌍의 목록을 입력받음
    #마지막 단계는 변환기/추정기 모두 사용 가능, 그외는 모두 변환기
    
    #파이프라인 하나만 실행시킬 경우
    housing_num_tr = num_pipline.fit_transform(housing_num)    #파이프라인의 fit_transform을 호출하면 모든 변환기의 fit_transform을 차례대로 호출

    cat_pipeline = Pipeline([
        ('cat_encoder',MyTransform_6.MyCategoricalEncoder(cat_attribs,encoding="onehot-dense")),
    ])

    #파이프라인 여러개 실행시킬 경우
    from sklearn.pipeline import FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipline",num_pipline),
        ("cat_pipeline",cat_pipeline)
    ])
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared.shape)