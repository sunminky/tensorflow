import pickle

a = [1,2,3]

# pickle로 저장하기
f = open("test.pkl", "wb")
pickle.dump(a, f)
f.close()

# pickle 불러오기
f = open("test.pkl", "rb")
temp = pickle.load(f)
print(temp)
f.close()