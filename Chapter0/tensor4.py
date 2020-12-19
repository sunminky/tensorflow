import numpy as np
#파이썬 이미지 처리 라이브러리 읽기
from PIL import Image

#이미지 파일 읽어 들이기
image = Image.open('./data/pict/sample.jpg','r')
#이미지 파일 출력하기
print(image)
#이미지 파일 픽셀값 추출하기
image_px = np.array(image)
#이미지 파일의 픽셀값 출력하기
print(image_px)

#이미지를 1차원 배열로 변환하기
image_flatten = image_px.flatten().astype(np.float32) / 255.0   #255로 나누어 정규화
print(image_flatten)
#이미지 픽셀값(배열)의 크기 출력하기
print(len(image_flatten))

#이미지를 그레이스케일로 변환하기
gray_image = image.convert('L')
print(gray_image)

#이미지 파일을 픽셀값으로 변환하기
gray_image_px = np.array(gray_image)
print(gray_image_px)

#이미지를 1차원 배열로 변환하기
gray_image_flatten = gray_image_px.flatten().astype(np.float32) / 255.0 #255로 나누어 정규화
print(len(gray_image_flatten))

'''충분한 이미지가 제공되지 못하는 경우 원래 이미지를 조금씩 가공해서 개수를 늘릴수 있다'''
##이미지 가공하기##
from PIL import ImageEnhance

#이미지 채도 조정하기
conv1 = ImageEnhance.Color(image)
conv1_image = conv1.enhance(0.5)    #이미지 채도를 계수 값(0.5)으로 조정, 0.0 : 검은 이미지 1.0 : 원래 이미지
conv1_image.save('./data/pict/sample_conv1.jpg')    #이미지 저장하기
print(conv1_image)

#이미지 명도 조정하기
conv2 = ImageEnhance.Brightness(image)
conv2_image = conv2.enhance(0.5)    #이미지 명도를 계수 값(0.5)로 조정, 0.0 : 검은 이미지 1.0 : 원래 이미지
conv2_image.save('./data/pict/sample_conv2.jpg')    #이미지 저장하기
print(conv2_image)

#이미지 콘트라스트 조정하기
conv3 = ImageEnhance.Contrast(image)
conv3_image = conv3.enhance(0.5)    #이미지 콘트라스트를 계수 값(0.5)로 조정, 0.0 : 검은 이미지 1.0 : 원래 이미지
conv3_image.save('./data/pict/sample_conv3.jpg')    #이미지 저장하기
print(conv3_image)

#이미지 날카로움 조정하기
conv4 = ImageEnhance.Sharpness(image)
conv4_image = conv4.enhance(2.0)    #이미지 날카로움을 계수 값(2.0)로 조정, 0.0 : 윤곽 흐림 1.0 : 원래 이미지 2.0 : 윤곽 강조(선명)
conv4_image.save('./data/pict/sample_conv4.jpg')    #이미지 저장하기
print(conv4_image)