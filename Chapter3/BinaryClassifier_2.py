import Chapter3.DownloadMNIST_1

X_train, X_test, y_train, y_test = Chapter3.DownloadMNIST_1.splitData()

y_train_5 = (y_train == 5)  #5는 True이고, 다른 숫자는 모두 False
y_test_5 = (y_test == 5)

## 숫자 5만을 식별하는 이진분류기 ##
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=500, random_state=42)  #매우 큰 데이터 셋을 효율적으로 처리, 온라인 학습에 유리
sgd_clf.fit(X_train,y_train_5)
result = sgd_clf.predict(X_test[0].reshape(1, -1))

print("예측결과 : ",result,"\n예측데이터 라벨 :",y_test[0])