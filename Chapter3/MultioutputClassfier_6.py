import Chapter3.BinaryClassifier_2 as ch3Bin

def multioutputClassfier():
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    noise = np.random.randint(0, 100, (len(ch3Bin.X_train), 784))
    X_train_mod = ch3Bin.X_train + noise    #원래 이미지에 노이즈 추가
    noise = np.random.randint(0, 100, (len(ch3Bin.X_test), 784))
    X_test_mod = ch3Bin.X_test + noise  #원래 이미지에 노이즈 추가
    y_train_mod = ch3Bin.X_train    #정답데이터는 노이즈 추가하기 전 데이터
    y_test_mod = ch3Bin.y_test  #정답데이터는 노이즈 추가하기 전 데이터

    knn_clf = KNeighborsClassifier()  # KNeighborsClassifier은 다중 레이블 분류를 지원
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict(X_test_mod[0].reshape(1, -1))

    import matplotlib.pyplot as plt

    def plot_digits(instances, images_per_row=10, **options):   #없어서 내가 만듬, 신경안써도 됨
        import matplotlib

        size = 28
        images_per_row = min(len(instances), images_per_row)
        images = [instance.reshape(size, size) for instance in instances]
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        n_empty = n_rows * images_per_row - len(instances)
        images.append(np.zeros((size, size * n_empty)))
        for row in range(n_rows):
            rimages = images[row * images_per_row: (row + 1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
        plt.imshow(image, cmap=matplotlib.cm.binary, **options)
        plt.axis("off")

    plot_digits(clean_digit, images_per_row=1)
    plt.show()
    print(y_test_mod[0], "복원")

if __name__ == '__main__':
    multioutputClassfier()