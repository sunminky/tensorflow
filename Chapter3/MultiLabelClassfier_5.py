import Chapter3.BinaryClassifier_2 as ch3Bin

def multiClassfier():
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    y_train_large = (ch3Bin.y_train >= 7)   #7이상의 수를 타겟으로 함
    y_train_odd = (ch3Bin.y_train % 2 == 1)    #홀수를 타겟으로 함
    y_multilabel = np.c_[y_train_large, y_train_odd]
    
    knn_clf = KNeighborsClassifier()    #KNeighborsClassifier은 다중 레이블 분류를 지원
    knn_clf.fit(ch3Bin.X_train, y_multilabel)
    result = knn_clf.predict(ch3Bin.X_test[0].reshape(1,-1))
    print(ch3Bin.y_test[0],"의 결과 :",result) #7 의 결과 : [[ True  True]]

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import f1_score

    ### 분류기 평가 ###
    y_train_knn_pred = cross_val_predict(knn_clf, ch3Bin.X_train, y_multilabel, cv=3, n_jobs=-1)    #모든 레이블에 대한 f1점수의 평균 계산, n_jobs=-1 cpu의 모든코어 사용
    result = f1_score(y_multilabel, y_train_knn_pred, average="macro")   #모든 레이블의 가중치가 같다고 가정
    print("클래스의 개수를 고려안한 f1 점수 :",result)
    result = f1_score(y_multilabel, y_train_knn_pred, average="weighted")  # 타겟 레이블에 속한 샘플의 수를 가중치로 줌
    print("클래스의 개수를 고려한 f1 점수 :", result)

if __name__ == '__main__':
    multiClassfier()