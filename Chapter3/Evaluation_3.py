import Chapter3.BinaryClassifier_2

def evaluation(sgd_clf):
    from sklearn.model_selection import cross_val_score
    
    result = cross_val_score(sgd_clf, Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train_5,    #k폴드 교차 검증 사용
                    cv=3, scoring="accuracy")   #3개의 서브셋으로 나눔, 정확도를 계산함(accuracy), 확률적경사하강법 적용
    print(result)   #[0.96135 0.96385 0.9533 ], 정확도를 성능지표로 사용하는 것은 불균형한 데이터셋(어떤 클래스가 다른 것보다 월등히 많음)에서 부정확

def confusionMatrix(sgd_clf):
    from sklearn.model_selection import cross_val_predict
    
    y_train_ped = cross_val_predict(sgd_clf, Chapter3.BinaryClassifier_2.X_train,   #각 테스트 폴드에서 얻은 예측 반환
                                    Chapter3.BinaryClassifier_2.y_train_5, cv=3)

    from sklearn.metrics import confusion_matrix

    confusion_result = confusion_matrix(Chapter3.BinaryClassifier_2.y_train_5, y_train_ped) #오차 행렬 생성
    print(confusion_result) #[[53670   909][ 1146  4275]], 오차 행렬 출력
    
def precisionANDrecall(sgd_clf):
    from sklearn.model_selection import cross_val_predict

    y_train_ped = cross_val_predict(sgd_clf, Chapter3.BinaryClassifier_2.X_train,  #각 테스트 폴드에서 얻은 예측 반환
                                    Chapter3.BinaryClassifier_2.y_train_5, cv=3)

    from sklearn.metrics import precision_score, recall_score
    
    print("정밀도 :", precision_score(Chapter3.BinaryClassifier_2.y_train_5, y_train_ped)) #정밀도 출력
    print("재현율 :", recall_score(Chapter3.BinaryClassifier_2.y_train_5, y_train_ped))    #재현율 출력
    
    from sklearn.metrics import f1_score
    
    print("f1점수 :", f1_score(Chapter3.BinaryClassifier_2.y_train_5, y_train_ped)) #f1점수 출력
    #f1점수 : F = 1 / ((a/정밀도) + ((1-a)/재현율)) = (b**2 + 1) * (정밀도 * 재현율 / ((b**2 * 정밀도) + 재현율)), b**2 = (1-a) / a
    #b가 1보다크면 재현율이 강조, 1보다 작으면 정밀도가 강조, b가 1일때 점수를 f1점수라고 함
    
def tradeoff(sgd_clf):
    y_scores = sgd_clf.decision_function(Chapter3.BinaryClassifier_2.X_train[0].reshape(1,-1))  #샘플의 점수를 반환
    print(Chapter3.BinaryClassifier_2.y_train[0], "의 점수 :", y_scores) #샘플점수 출력

    threshhold = 0
    y_some_digit_pred = (y_scores > threshhold)
    print(y_some_digit_pred)    #임계값보다 높으면 true, 작으면 false

    threshhold = 20000  #임계값 증가
    y_some_digit_pred = (y_scores > threshhold)
    print(y_some_digit_pred)  # 임계값보다 높으면 true, 작으면 false

    from sklearn.model_selection import cross_val_predict

    y_scores = cross_val_predict(sgd_clf, Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train_5,
                                 cv=3, method="decision_function")  #훈련세트에 있는 모든 샘플의 점수를 구함
    
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(Chapter3.BinaryClassifier_2.y_train_5, y_scores)  #가능한 모든 임곗값에 대한 정밀도와 재현율 계산
    
    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):    #임계값에 대한 정밀도와 재현율 시각화
        import matplotlib.pyplot as plt

        plt.plot(thresholds, precisions[:-1], "b--", label="precision")
        plt.plot(thresholds, recalls[:-1], "g--", label="recalls")
        plt.xlabel("thresholds")
        plt.legend(loc="center left")
        plt.ylim([0, 1])
        plt.show()

    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    def plot_precision_recall_and_threshold(precisions, recalls):    #정밀도와 재현율의 관계를 시각화
        import matplotlib.pyplot as plt

        #재현율이 변화할때 정밀도의 변화 확인
        plt.title("Relation between precision and recall")
        plt.plot(recalls[:-1], precisions[:-1], "b--", label="precision")
        plt.xlabel("recalls")
        plt.ylabel("precisions")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

    plot_precision_recall_and_threshold(precisions, recalls)
    
    y_train_90 = (y_scores > 3000)  #정밀도가 90이상이 되려면 임계값이 5000이상이여야 함

    from sklearn.metrics import precision_score, recall_score

    #재현율이 너무 낮으면 정밀도가 높아도 유용하지 않음
    print("정밀도 :", precision_score(Chapter3.BinaryClassifier_2.y_train_5, y_train_90))  # 정밀도 출력
    print("재현율 :", recall_score(Chapter3.BinaryClassifier_2.y_train_5, y_train_90))  # 재현율 출력

    ### PR곡선을 사용하는 경우 ###
    # 1. 양성 클래스가 드뭄
    # 2. 거짓 음성보다 거짓 양성이 다 중요

def rocCurve(sgd_clf):
    from sklearn.model_selection import cross_val_predict

    y_scores = cross_val_predict(sgd_clf, Chapter3.BinaryClassifier_2.X_train, Chapter3.BinaryClassifier_2.y_train_5,
                                 cv=3, method="decision_function")  # 훈련세트에 있는 모든 샘플의 점수를 구함

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(Chapter3.BinaryClassifier_2.y_train_5, y_scores)   #거짓양성비율, 진짜양성비율, 임계값 계산

    def plot_roc_curve(fpr, tpr, label=None):   #roc곡선 시각화
        from matplotlib import pyplot as plt

        #roc곡선아래 넓이(AUC)가 1에 가까울수록 좋은 곡선
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], "k--") #완전한 랜덤분류기(찍어맞추기)의 roc곡선 의미, 좋은 분류기는 이 직선으로부터 최대한 멀리 떨어져야함
        plt.axis([0, 1, 0, 1])  #x, y축의 범위설정, axis([xmin, xmax, ymin, ymax])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        compareRndForest(fpr, tpr, "RandomForest")
        plt.legend(loc="lower right")
        plt.show()
        
    def compareRndForest(fpr, tpr, label=None): #랜덤포레스트의 ROC 시각화
        from sklearn.ensemble import RandomForestClassifier

        forest_clf = RandomForestClassifier(random_state=42)
        y_probas_forest = cross_val_predict(forest_clf, Chapter3.BinaryClassifier_2.X_train,
                                            Chapter3.BinaryClassifier_2.y_train_5, cv=3,
                                            method="predict_proba")
        #랜덤포레스트분류기는 decision_function대신 샘플이 주어진 클래스에 있을 확률을 반환하는 predict_proba 사용
        y_scores_forest = y_probas_forest[:, 1] #양성 클래스에 대한 확률을 점수로 사용
        fpr_forest, tpr_forest, thresholds_forest = roc_curve(Chapter3.BinaryClassifier_2.y_train_5, y_scores_forest)

        from matplotlib import pyplot as plt

        plt.plot(fpr_forest, tpr_forest, label=label)

        from sklearn.metrics import roc_auc_score

        print("RandomForest's AUC :", roc_auc_score(Chapter3.BinaryClassifier_2.y_train_5, y_scores_forest))  # Roc의 AUC반환

    plot_roc_curve(fpr, tpr, "SGD")
    
    from sklearn.metrics import roc_auc_score
    
    print("SGD's AUC :", roc_auc_score(Chapter3.BinaryClassifier_2.y_train_5, y_scores))  #Roc의 AUC반환


if __name__ == '__main__':
    sgd_clf = Chapter3.BinaryClassifier_2.getBinaryClassifier()
    #evaluation(sgd_clf)
    #confusionMatrix(sgd_clf)
    #precisionANDrecall(sgd_clf)
    #tradeoff(sgd_clf)
    rocCurve(sgd_clf)