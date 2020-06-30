import numpy as np
import pandas as pd


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))  # 0 ~ len(data)까지 무작위로 섞은 배열 반환, shuffle과는 다르게 원본 훼손 없음
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]  # 랜덤으로 배열을 나눠서 반환, 실행할때마다 반환하는 데이터가 달라지는 문제있음


from zlib import crc32


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32  # 해쉬의 마지막값이 특정 값보다 작으면 True, 아니면 False


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]  # id_column에 해당하는 열
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))  # id_는 ids 그자체가 넘어감, apply 함수는 모든 행에 함수를 적용시킴
    return data.loc[~in_test_set], data.loc[in_test_set]


if __name__ == '__main__':
    data = pd.DataFrame([{"c1": 26, "c2": 12}, {"c1": 29, "c2": 12}, {"c1": 16, "c2": 12}, {"c1": 33, "c2": 26}, {"c1": 2, "c2": 26}])
    a, b = split_train_test(data.reset_index(), 0.75)  # 학습용, 훈련용 순서로 반환
    print(a)
    print(b)
    print("----------------------------------------")
    a, b = split_train_test_by_id(data.reset_index(), 0.75, "index")  # 학습용, 훈련용 순서로 반환
    print(a)
    print(b)

    # 우리가 만든 함수랑 비슷한 동작을 하는 기본제공 함수
    from sklearn.model_selection import train_test_split

    a, b = train_test_split(data, test_size=0.75, random_state=42)  # 나누는 비율 0.75, 시드값 42
    print("----------------------------------------")
    print(a)
    print(b)

    # 우리가 만든 함수랑 비슷한 동작을 하는 기본제공 함수2, 데이터가 편향되지 않도록 해줌
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                   random_state=42)  # 세트의 개수 1개(K폴드 알고리즘), 나누는 비율 0.5, 시드값 42
    print("----------------------------------------")
    for train_index, test_index in split.split(data, data["c2"]):   #c2행의 비율을 고려해서 나눔
        a = data.loc[train_index]  # 인덱스를 기준으로 행을 읽기, iloc은 행번호를 기준으로 행을 읽음
        b = data.loc[test_index]
        print(a)
        print(b)
