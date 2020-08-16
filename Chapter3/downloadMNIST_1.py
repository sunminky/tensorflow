from sklearn.datasets import fetch_openml
import pickle

try:
    ## 이미 한번 다운받은 적이 있다면 생략 ##
    with open("mnist.pkl","rb") as f:
        mnist = pickle.load(f)
except FileNotFoundError:
    mnist = fetch_openml('mnist_784', version=1)    #MNIST 데이터 다운로드
    with open("mnist.pkl","wb") as f:
        pickle.dump(mnist,f)

X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

## MNIST 데이터 시각화 ##
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)    # 가로 28 세로 28 이미지 배열로 만듬
plt.imshow(some_digit_image, cmap='gray', interpolation = "nearest")
plt.axis("off") #가로축, 세로축 없앰
plt.show()
print(y[36000]) #데이터의 레이블 확인

## 학습데이터, 훈련데이터 분리 ##
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
## 데이터 순서 섞기 ##
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]