import numpy as np

list_data = [1, 2, 3]
array = np.array(list_data)

print("배열 타입 :",array.dtype)
print("배열 크기 :",array.size)

array1 = np.arange(4)   #0~3까지 배열 만들기
array2 = np.zeros((4,4), dtype=np.float)    #4x4 크기의 0으로 초기화된 배열
array3 = np.ones((1,4), dtype=np.str)   #1x4 크기의 1로 초기화된 배열
array4 = np.random.randint(0,10,(3,3))  #0~9사이의 랜덤한 값이 들어감 3x3배열
array5 = np.random.normal(0,1,(3,3))    #평균이 0이고, 표준편차가 1인 표준정규분포를 따르는 3x3배열
array4_5 = np.concatenate([array4,array5], axis=0)  #배열4와 배열5를 행을 기준으로 합침, axis=1이면 열 기준으로 합침
array2_r = array2.reshape((2,8))    #4x4 배열을 2x8 배열로 바꿈
array2_s = np.split(array2,[3],axis=1)  #배열을 3열 기준으로 나눔
array2_3 = array2 + np.array(array3, dtype=np.float)*3  #4x4 배열과 1x4 배열을 더함(브로드캐스트), array3의 값을 3배로 만들어서 더함
array4_m = array4 < 4   #원소중 4보다 작은값은 False로 아닌것은 True로 바꿈(마스킹)
array4[array4_m] = -1   #원소중 4보다 작은값은 -1로 바꿈

###넘파이 저장###
saveArr1 = np.random.randint(0,10,size=9).reshape((3,3))
saveArr2 = np.random.randint(0,10,size=9).reshape((3,3))
np.save("saved.npy",saveArr1)    #넘파이 객체 저장
loadArr1 = np.load("saved.npy")  #저장된 객체 로드
np.savez("saved.npz", saveArr1=saveArr1, saveArr2=saveArr2) #복수개의 객체 저장
result = np.load("saved.npz")   #복수개의 객체 로드
loadArr1 = result["saveArr1"]   #배열의 이름을 키로 추출
loadArr2 = result["saveArr2"]   #배열의 이름을 키로 추출

##넘파이 정렬##
seqArr = np.array([[5, 9, 10, 3, 1],[8, 3, 4, 2, 5]])
seqArr.sort(axis=0) #열을 기준으로 정렬
print(np.unique(seqArr))   #중복된 원소 제거