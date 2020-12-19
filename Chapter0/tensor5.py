##텐서플로 라이브러리##
import tensorflow as tf

##tflearn라이브러리##
import tflearn

##레이어 생성 등 학습에 필요한 라이브버리 읽어 들이기##
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import os
import numpy as np
from PIL import Image

##이미지 데이터 처리하기

#학습 전용 이미지 파일을 저장하고 있는 디렉터리
train_dirs = ['pos','neg']

#학습 데이터를 저장할 배열 준비하기
trainX = [] #이미지 픽셀값
trainY = [] #정답 데이터

for i,d in enumerate(train_dirs):
    #파일 이름 추출하기
    files = os.listdir('./data/pict/'+d)
    for f in files:
        #이미지 읽어 들이기
        image = Image.open('./data/pict/'+d+'/'+f,'r')
        #그레이스케일로 변환
        gray_image = image.convert('L')
        #이미지 파일을 픽셀값으로 변환하기
        gray_image_px = np.array(gray_image)
        gray_image_flatten = gray_image_px.flatten().astype(np.float32) / 255.0
        trainX.append(gray_image_flatten)

        #정답 데이터를 one_hot 형식으로 변환하기
        tmp = np.zeros(2)
        tmp[i] = 1
        trainY.append(tmp)  #정답데이터 만들기

#numpy 배열로 변환하기
trainX = np.array(trainX)
trainY = np.array(trainY)

#이미지 픽셀 데이터를 2차원으로 변환하기
trainX = trainX.reshape([-1,32,32,1])   #CNN을 사용해 학습하려면 학습 데이터가 2차원이여 함

##신경망 만들기##
#초기화
tf.reset_default_graph()

#입력 레이어 만들기
net = tflearn.input_data(shape=[None,32,32,1]) #입력할 학습데이터 형태 : None 이미지 크기 : 32*32 그레이스케일 지정

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
net = tflearn.fully_connected(net,2,activation='softmax')  #생성할 레이어의 바로 앞 레이어 : net 생성할 레이어 노드수 : 2 사용할 활성화 함수 : softmax
net = tflearn.regression(net,optimizer='sgd',learning_rate=0.5,loss='categorical_crossentropy')
#학습 조건 설정, 학습대상 레이어 : net 최적화 방법 : sgd(확률적 경사 하강법) 학습계수 : 0.5 오차함수 : categorical_crossentropy(교차엔트로피)

##모델만들기##
#학습하기
model = tflearn.DNN(net)    #생성한 신경망과 학습조건 설정, 대상신경망 : net
model.fit(trainX,trainY,n_epoch=20,batch_size=100,validation_set=0.1,show_metric=True)  #학습을 실행하고 모델을 생성
#학습데이터 : trainX 정답 데이터 : trainY 학습획수 : 20 배치 한번 당 주는 데이터 샘플의 크기 : 100 모델 정밀도를 검증하기 위한 테스트 세트 : 0.1(10%) 학습 단계별로 정밀도 출력 : True
