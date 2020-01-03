##텐서플로 라이브러리##
import tensorflow as tf

##tflearn라이브러리##
import tflearn

##MNIST 데이터 세트를 다루기 위한 라이브러리##
import tflearn.datasets.mnist as mnist

import numpy as np

##MNIST 데이터를 ./data/mnist에 내려받고, 압축을 해제한 다음 각 변수에 할당하기
trainX, trainY, testX, testY = mnist.load_data('./data/mnist',one_hot=True)
#정답 데이터를 one-hot 형식으로 하겠다. one hot은 "정답의 종류"를 크기로하는 배열에서 정답을 1로지정하고 나머지는 0으로 지정

#이미지 픽셀 데이터를 1차원에서 2차원으로 변경하기
trainX = np.reshape(trainX,(-1,28,28))   #28*28로 변경

##신경망 만들기##
#초기화
tf.reset_default_graph()

#입력 레이어 만들기
net = tflearn.input_data(shape=[None,28,28]) #입력할 학습데이터 형태 : None 이미지 크기 : 28*28

##중간레이어 만들기
#LSTM 블록
net = tflearn.lstm(net,128)

#출력 레이어 만들기
net = tflearn.fully_connected(net,10,activation='softmax')  #생성할 레이어의 바로 앞 레이어 : net 생성할 레이어 노드수 : 10 사용할 활성화 함수 : softmax
net = tflearn.regression(net,optimizer='sgd',learning_rate=0.5,loss='categorical_crossentropy')
#학습 조건 설정, 학습대상 레이어 : net 최적화 방법 : sgd(확률적 경사 하강법) 학습계수 : 0.5 오차함수 : categorical_crossentropy(교차엔트로피)

##모델만들기##
#학습하기
model = tflearn.DNN(net)    #생성한 신경망과 학습조건 설정, 대상신경망 : net
model.fit(trainX,trainY,n_epoch=20,batch_size=100,validation_set=0.1,show_metric=True)  #학습을 실행하고 모델을 생성
#학습데이터 : trainX 정답 데이터 : trainY 학습획수 : 20 배치 한번 당 주는 데이터 샘플의 크기 : 100 모델 정밀도를 검증하기 위한 테스트 세트 : 0.1(10%) 학습 단계별로 정밀도 출력 : True