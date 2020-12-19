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

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                   random_state=42)  # 세트의 개수 1개(K폴드 알고리즘), 나누는 비율 0.5, 시드값 42
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
        ('selector', MyTransform_6.DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', MyTransform_6.CombinedAttributesAdder()),
        ('stdScaler', StandardScaler()),
    ])  # Pipeline은 연속된 단계를 나타내는 이름/추정기 쌍의 목록을 입력받음
    # 마지막 단계는 변환기/추정기 모두 사용 가능, 그외는 모두 변환기

    # 파이프라인 하나만 실행시킬 경우
    housing_num_tr = num_pipline.fit_transform(housing_num)  # 파이프라인의 fit_transform을 호출하면 모든 변환기의 fit_transform을 차례대로 호출

    cat_pipeline = Pipeline([
        ('cat_encoder', MyTransform_6.MyCategoricalEncoder(cat_attribs, encoding="onehot-dense")),
    ])

    # 파이프라인 여러개 실행시킬 경우
    from sklearn.pipeline import FeatureUnion

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipline", num_pipline),
        ("cat_pipeline", cat_pipeline)
    ])
    housing_prepared = full_pipeline.fit_transform(housing)

    ### 예측 ###
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared,housing_labels)    #선형회귀 모델 훈현

    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared,housing_labels)   #결정트리 모델 훈련
    
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared,housing_labels) #랜덤포레스트 모델(앙상블) 훈련

    ###테스트 적용###
    from sklearn.metrics import mean_squared_error
    housing_predict = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predict)   #오차 계산
    lin_rmse = np.sqrt(lin_mse)
    print("예측오차 :",lin_rmse) #예측 오차

    ###K폴드 교차검증###
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    #10개의 서브셋으로 나눔, 평균제곱오차의 반댓값을 계산함(neg_mean_squared_error), 결정트리 모델 적용
    tree_rmse_scores = np.sqrt(-scores) #사이킷런 교차검증이 효용 함수를 기대하기 때문에 음수로 바꿈
    print("스코어 :",tree_rmse_scores)
    print("평균 :",tree_rmse_scores.mean())
    print("표준 편차 :",tree_rmse_scores.std())

    ###하이퍼파라메터 튜닝###
    ######그리드 탐색######
    '''비교적 적은 수의 조합을 탐구할 때 추천'''
    from sklearn.model_selection import GridSearchCV

    parm_grid = [
        {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},   #모형의 개수 [3,10,30], 다차원 독립 변수 중 선택할 차원의 수 [2,4,6,8]
        {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]},   #데이터 중복사용 :False
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, parm_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)   # (3x4 + 2x3)x5개의 조합 시도 , 모형개수와 차원의 수와 kfold 세트 개수의 곱인 모든 조합을 시도함

    print(grid_search.best_params_) #(탐색값 범위내에서) 획득한 최적의 하이퍼파라메터
    print(grid_search.best_estimator_)  #직접 추정기에 접근해서 파라메터 관찰

    ##각 파라메터별 평가점수 확인##
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    ###랜덤탐색###
    '''하이퍼파라메터의 탐색공간이 클 경우 사용'''
    from sklearn.model_selection import RandomizedSearchCV

    parm_random = [
        {'n_estimators': list(range(1,50,5)), 'max_features': [2, 4, 6, 8, 10, 12]},
        # 모형의 개수 [3,10,30], 다차원 독립 변수 중 선택할 차원의 수 [2,4,6,8]
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},  # 데이터 중복사용 :False
    ]
    forest_reg = RandomForestRegressor()
    random_search = RandomizedSearchCV(forest_reg, parm_random, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True,n_iter=10)   #최대시도 횟수 10으로 제한
    random_search.fit(housing_prepared, housing_labels)

    print(random_search.best_params_)  # (탐색값 범위내에서) 획득한 최적의 하이퍼파라메터
    print(random_search.best_estimator_)  # 직접 추정기에 접근해서 파라메터 관찰

    ###최상의 모델 오차분석###
    feature_importances = grid_search.best_estimator_.feature_importances_  #각 특성의 상대적인 중요도
    print(feature_importances)

    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_one_hot_attribs = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
    attributes = list(num_attribs) + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))   #중요도와 그에 대응하는 특성 표시

    ###시스템 평가###
    final_model = grid_search.best_estimator_  #최적의 모델 선택
    X_test = start_test_set.drop("median_house_value", axis=1)
    y_test = start_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.fit_transform(X_test)

    final_predictions = final_model.predict(X_test_prepared)
    
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse) #시스템 오차 출력