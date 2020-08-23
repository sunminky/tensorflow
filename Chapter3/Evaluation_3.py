import Chapter3.BinaryClassifier_2

def evaluation(sgd_clf):
    from sklearn.model_selection import cross_val_score
    
    result = cross_val_score(sgd_clf, Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train_5,    #k폴드 교차 검증 사용
                    cv=3, scoring="accuracy")   #3개의 서브셋으로 나눔, 정확도를 계산함(accuracy), 확률적경사하강법 적용
    print(result)   #[0.96135 0.96385 0.9533 ], 정확도를 성능지표로 사용하는 것은 불균형한 데이터셋(어떤 클래스가 다른 것보다 월등히 많음)에서 부정확

def confusionMatrix(sgd_clf):
    from sklearn.model_selection import cross_val_predict
    
    y_train_ped = cross_val_predict(sgd_clf, Chapter3.BinaryClassifier_2.X_train,   #k폴드 교차 검증수행, 각 테스트 폴드에서 얻은 예측 반환
                                    Chapter3.BinaryClassifier_2.y_train_5, cv=3)

    from sklearn.metrics import confusion_matrix

    confusion_result = confusion_matrix(Chapter3.BinaryClassifier_2.y_train_5, y_train_ped) #오차 행렬 생성
    print(confusion_result) #[[53670   909][ 1146  4275]], 오차 행렬 출력
    
    pass

if __name__ == '__main__':
    sgd_clf = Chapter3.BinaryClassifier_2.getBinaryClassifier()
    #evaluation(sgd_clf)
    confusionMatrix(sgd_clf)