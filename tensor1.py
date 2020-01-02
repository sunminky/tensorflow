import tensorflow as tf

xData = [1,2,3,4,5,6,7]
yData = [25000,55000,75000,110000,128000,155000,180000]
w = tf.Variable(tf.random_uniform([1],-100,100))   #가설의 기울기, (=가중치)
b = tf.Variable(tf.random_uniform([1],-100,100))    #y절편,(=바이어스)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
h = w * x + b   #가설 설정
cost = tf.reduce_mean(tf.square(h-y))      #비용함수, 실제값과 예측값 거리의 오차의 평균값을 구함
a = tf.Variable(0.01)   #경사하강 알고리즘에서 얼마만큼 점프할것인가
oprimizer = tf.train.GradientDescentOptimizer(a)    #경사하강 라이브러리
train = oprimizer.minimize(cost)        #비용함수를 가장 적게 만드는 방향으로 학습
init = tf.global_variables_initializer()    #텐서플로우에서 쓸 변수 초기화
session = tf.Session()      #텐서플로우에서 세션을 얻어옴
session.run(init)   #세션 초기화

'''실제 학습을 진행하는 구간'''
for i in range(5001):
    session.run(train,feed_dict={x : xData, y : yData}) #실제로 학습 진행
    if i % 500 == 0:
        print(i,session.run(cost,feed_dict={x : xData,y : yData}),session.run(w),session.run(b))
        #i 출력, 학습값 출력, w값 출력, b값 출력
'''학습 끝'''

print(session.run(h,feed_dict={x:[8]}))    #원하는 입력(8)에 대한 결과 출력