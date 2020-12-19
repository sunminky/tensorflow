##텐서플로 라이브러리##
import tensorflow as tf

##tflearn라이브러리##
import tflearn

##레이어 생성 등 학습에 필요한 라이브버리 읽어 들이기##
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

##MNIST 데이터 세트를 다루기 위한 라이브러리##
import tflearn.datasets.mnist as mnist

import numpy as np

##MNIST 데이터를 ./data/mnist에 내려받고, 압축을 해제한 다음 각 변수에 할당하기
trainX, trainY, testX, testY = mnist.load_data('./data/mnist',one_hot=True)
#정답 데이터를 one-hot 형식으로 하겠다. one hot은 "정답의 종류"를 크기로하는 배열에서 정답을 1로지정하고 나머지는 0으로 지정

#학습 전용 이미지 픽셀 데이터와 정답 데이터의 크기 확인하기
print(len(trainX),len(trainY))

#테스트 전용 이미지 픽셀 데이터와 정답 데이터의 크기 확인하기
print(len(testX),len(testY))

#이미지 픽셀 데이터를 1차원에서 2차원으로 변경하기
trainX = trainX.reshape([-1,28,28,1])   #28*28로 변경, 그레이스케일 이미지이므로 컬러 채널 1로 설정
testX = testX.reshape([-1,28,28,1])     #28*28로 변경, 그레이스케일 이미지이므로 컬러 채널 1로 설정

#변환된 데이터 확인하기
print(trainX[0])
print(len(trainX[0]))

##신경망 만들기##
#초기화
tf.reset_default_graph()

#입력 레이어 만들기
net = tflearn.input_data(shape=[None,28,28,1]) #입력할 학습데이터 형태 : None 이미지 크기 : 28*28 그레이스케일 지정

##중간 레이어 만들기
#합성곱 레이어 만들기
net = conv_2d(net,32,5,activation='relu')   #생성할 레이어의 바로 앞 레이어 : net 합성곱 필터 수(출력 차원 수) : 32 필터크기 : 5*5 사용할 활성화 함수 : relu
#풀링 레이어 만들기
net = max_pool_2d(net,2)    #생성할 레이어의 바로 앞 레이어 : net 최대 풀링 영역 : 2*2
#합성곱 레이어
net = conv_2d(net,64,5,activation='relu')   #생성할 레이어의 바로 앞 레이어 : net 합성곱 필터 수(출력 차원 수) : 64 필터크기 : 5*5 사용할 활성화 함수 : relu
#풀링 레이어 만들기
net = max_pool_2d(net,2)    #생성할 레이어의 바로 앞 레이어 : net 최대 풀링 영역 : 2*2
#전결합 레이어 만들기
net = tflearn.fully_connected(net,128,activation='relu')    #생성할 레이어의 바로 앞 레이어 : net 생성할 레이어 노드수 : 128 사용할 활성화 함수 : relu
net = tflearn.dropout(net,0.5)  #드롭아웃 대상 레이어 : net  노드 값을 줄일 비율 : 0.5


#출력 레이어 만들기
net = tflearn.fully_connected(net,10,activation='softmax')  #생성할 레이어의 바로 앞 레이어 : net 생성할 레이어 노드수 : 10 사용할 활성화 함수 : softmax
net = tflearn.regression(net,optimizer='sgd',learning_rate=0.5,loss='categorical_crossentropy')
#학습 조건 설정, 학습대상 레이어 : net 최적화 방법 : sgd(확률적 경사 하강법) 학습계수 : 0.5 오차함수 : categorical_crossentropy(교차엔트로피)

##모델만들기##
#학습하기
model = tflearn.DNN(net)    #생성한 신경망과 학습조건 설정, 대상신경망 : net
model.fit(trainX,trainY,n_epoch=20,batch_size=100,validation_set=0.1,show_metric=True)  #학습을 실행하고 모델을 생성
#학습데이터 : trainX 정답 데이터 : trainY 학습획수 : 20 배치 한번 당 주는 데이터 샘플의 크기 : 100 모델 정밀도를 검증하기 위한 테스트 세트 : 0.1(10%) 학습 단계별로 정밀도 출력 : True

##모델 적용(예측)##
pred = np.array(model.predict(testX)).argmax(axis=1)    #모델을 testX에 적용하고 결과를 pred에 저장
#argmax : 최대값의 인데스를 구함
print(pred)

label = testY.argmax(axis=1)    #정답 값을 저장하고 있는 배열을 label에 저장
print(label)

accuracy = np.mean(pred == label, axis=0)   #예측값과 정답이 얼마나 같은지 출력(정밀도 출력)
print(accuracy)


