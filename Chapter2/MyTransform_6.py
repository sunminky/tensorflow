from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

room_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_household = X[:, bedrooms_ix] / X[:, household_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_household]  #두개의 배열을 합침, c_ : 열 확장, r_ : 행 확장
        else:
            return np.c_[X, rooms_per_household, population_per_household]  # 두개의 배열을 합침, c_ : 열 확장, r_ : 행 확장

### 사용법 ###
'''attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.fit_transform(housing.value)'''

#판다스 프레임을 넘파일 배열로 바꾸는 클래스
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values

#카레고리 데이터를 원핫인코딩으로 바꾸는 클래스
class MyCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_name, encoding="onehot-dense"):
        self.encoding = encoding
        self.attribute_name = attribute_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        housing_tr = X[self.attribute_name]
        housing_cat_encoded, housing_categories = housing_tr.factorize()  # 각 카테고리 정수값으로 매칭,

        # 정수값으로 매칭된 값을 원핫인코딩
        encoder = OneHotEncoder()
        housing_cat_1hot = encoder.fit_transform(
            housing_cat_encoded.reshape(-1, 1))  # 원핫인코딩, 입력으로 2차월 배열을 넣어야 함, 출력으로 희소행렬 리턴
        return housing_cat_1hot.toarray()  # 희소행렬(사이파이)을 밀집행렬(넘파이)로 반환