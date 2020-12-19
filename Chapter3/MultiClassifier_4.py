import Chapter3.BinaryClassifier_2


# 일대다(OvA)전략, 구분해야할 숫자가 10개라면 10개의 분류기를 훈련시켜 결정점수가 가장높은 것을 클래스로 선택
# 일대일(OvO)전략, 구분해야할 숫자가 10개라면 모든 수의 조합((0,1), (0,2)..)마다 이진분류기를 훈련, 45개의 분류가 필요

def multiClassfier(sgd_clf):
    # 다중 클래스 분류에 이진 분류 알고리즘을 선택하면 사이킷런이 자동으로 OvA적용, SVM분류기일 경우 OvO적용
    sgd_clf.fit(Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train)  # 분류기들 훈련
    result = sgd_clf.predict(Chapter3.BinaryClassifier_2.X_test[0].reshape(1, -1))
    print(Chapter3.BinaryClassifier_2.y_test[0], "의 예측결과 :", result)
    some_digit_scores = sgd_clf.decision_function(
        Chapter3.BinaryClassifier_2.X_train[0].reshape(1, -1))  # 각 클래스별 결정점수 반환
    print("---클래스별 점수---\n", some_digit_scores)

    import numpy as np
    print("점수가 가장 큰 클래스 :", np.argmax(some_digit_scores))  # 점수가 가장 큰 클래스 출력
    print(sgd_clf.classes_)  # 클래스 속성들 출력


def forceStrategy():
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsOneClassifier

    ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=500, random_state=42))  # OvO모델을 쓰도록 강제함
    ovo_clf.fit(Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train)
    result = ovo_clf.predict(Chapter3.BinaryClassifier_2.X_test[0].reshape(1, -1))
    print(Chapter3.BinaryClassifier_2.y_test[0], "의 OvO예측결과 :", result)
    print("OvO 분류기 개수 :", len(ovo_clf.estimators_))  # OvO 분류기 개수 : 45

    from sklearn.multiclass import OneVsRestClassifier

    ova_clf = OneVsRestClassifier(SGDClassifier(max_iter=500, random_state=42))  # OvA모델을 쓰도록 강제함
    ova_clf.fit(Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train)
    result = ova_clf.predict(Chapter3.BinaryClassifier_2.X_test[0].reshape(1, -1))
    print(Chapter3.BinaryClassifier_2.y_test[0], "의 OvO예측결과 :", result)
    print("OvA 분류기 개수 :", len(ova_clf.estimators_))  # OvA 분류기 개수 : 10

    from sklearn.ensemble import RandomForestClassifier

    forest_clf = RandomForestClassifier(random_state=42)  # 랜던포레스트분류기는 직접 샘플을 다중클래스로 분류가능, OvO나 OvA 적용할 필요없음
    forest_clf.fit(Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train)  # 랜덤포레스트분류기 훈련
    result = forest_clf.predict(Chapter3.BinaryClassifier_2.X_test[0].reshape(1, -1))
    print(Chapter3.BinaryClassifier_2.y_test[0], "의 랜덤포레스트 예측결과 :", result)  # 7 의 랜덤포레스트 예측결과 : [7]
    print("클래스별 예측확률 :", forest_clf.predict_proba(
        Chapter3.BinaryClassifier_2.X_test[0].reshape(1, -1)))  # 클래스별 예측확률 : [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]


def estimateAccuracy(sgd_clf):
    sgd_clf.max_iter = 1500

    from sklearn.model_selection import cross_val_score

    result = cross_val_score(sgd_clf, Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train, # k폴드 교차 검증 사용
                             cv=3, scoring="accuracy")  # 3개의 서브셋으로 나눔, 정확도를 계산함(accuracy), 확률적경사하강법 적용
    print("정확도 :", result)  #정확도 : [0.86375 0.8897  0.875  ]

    ## 데이터가 정규분포를 이루게 하여 정확도 높이기 ###
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    scaler = StandardScaler()  # 평균이 0과 표준편차가 1인 정규분포를 따르도록 변환
    X_train_scaled = scaler.fit_transform(Chapter3.BinaryClassifier_2.X_train.astype(np.float64))
    result = cross_val_score(sgd_clf, X_train_scaled, Chapter3.BinaryClassifier_2.y_train,  # k폴드 교차 검증 사용
                             cv=3, scoring="accuracy")  # 3개의 서브셋으로 나눔, 정확도를 계산함(accuracy), 확률적경사하강법 적용
    print("정확도 :", result)  #정확도 : [0.90425 0.89815 0.90025]

def reviewError(sgd_clf):
    ## 데이터가 정규분포를 이루게 하여 정확도 높이기 ###
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    scaler = StandardScaler()  # 평균이 0과 표준편차가 1인 정규분포를 따르도록 변환
    X_train_scaled = scaler.fit_transform(Chapter3.BinaryClassifier_2.X_train.astype(np.float64))
    
    from sklearn.model_selection import cross_val_predict

    y_train_ped = cross_val_predict(sgd_clf, Chapter3.BinaryClassifier_2.X_train,  # 각 테스트 폴드에서 얻은 예측 반환
                                    Chapter3.BinaryClassifier_2.y_train, cv=3, n_jobs=-1)
    
    from sklearn.metrics import confusion_matrix
    
    conf_mx = confusion_matrix(Chapter3.BinaryClassifier_2.y_train, y_train_ped)    #오차행렬 생성
    print("오차행렬 출력\n", conf_mx)

    import matplotlib.pyplot as plt

    plt.matshow(conf_mx, cmap=plt.cm.gray)  #2차원 배열형태의 숫자 데이터를 히트맵으로 표시
    plt.show()  #색깔이 밝을수록 값이 높음

    row_sums = conf_mx.sum(axis=1, keepdims=True)   #행 단위합 구하기, sum 이후에도 행렬 차원을 유지
    norm_conf_mx = conf_mx / row_sums   #각 좌표의 값 / 대응되는 클래스의 이미지개수 = 에러 비율
    np.fill_diagonal(norm_conf_mx, 0)   #주대각선(알맞게 분류한 샘플개수)을 0으로 채움
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray) #2차원 배열형태의 숫자 데이터를 히트맵으로 표시
    plt.show()

    def plot_digits(instances, images_per_row=10, **options):   #없어서 내가 만듬, 신경안써도 됨
        import matplotlib

        size = 28
        images_per_row = min(len(instances), images_per_row)
        images = [instance.reshape(size, size) for instance in instances]
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        n_empty = n_rows * images_per_row - len(instances)
        images.append(np.zeros((size, size * n_empty)))
        for row in range(n_rows):
            rimages = images[row * images_per_row: (row + 1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
        plt.imshow(image, cmap=matplotlib.cm.binary, **options)
        plt.axis("off")

    ## 샘플 출력 ##
    cl_a, cl_b = 2, 8
    #정답데이터가 2, 예측결과가 2
    X_aa = Chapter3.BinaryClassifier_2.X_train[(Chapter3.BinaryClassifier_2.y_train == cl_a) & (y_train_ped == cl_a)]
    # 정답데이터가 2, 예측결과가 8
    X_ab = Chapter3.BinaryClassifier_2.X_train[(Chapter3.BinaryClassifier_2.y_train == cl_a) & (y_train_ped == cl_b)]
    # 정답데이터가 8, 예측결과가 2
    X_ba = Chapter3.BinaryClassifier_2.X_train[(Chapter3.BinaryClassifier_2.y_train == cl_b) & (y_train_ped == cl_a)]
    # 정답데이터가 8, 예측결과가 8
    X_bb = Chapter3.BinaryClassifier_2.X_train[(Chapter3.BinaryClassifier_2.y_train == cl_b) & (y_train_ped == cl_b)]

    plt.figure(figsize=(8,8))
    plt.subplot(221);   plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222);   plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223);   plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224);   plot_digits(X_bb[:25], images_per_row=5)
    plt.show()
    
    #에측정확성을 높이기 위해 글자가 회전되지 않고 중앙이 위치하도록 전처리할 필요가 있다

    
if __name__ == '__main__':
    sgd_clf = Chapter3.BinaryClassifier_2.getBinaryClassifier()
    # multiClassfier(sgd_clf)
    # forceStrategy()
    #estimateAccuracy(sgd_clf)
    reviewError(sgd_clf)
