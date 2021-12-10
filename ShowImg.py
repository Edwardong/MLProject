import utils.mnist_reader as um
import numpy as np
import matplotlib.pyplot as plt
import LoadData as load

X_train, y_train, X_test, y_test = load.Load()
Xtr=X_train/255
Xte=X_test/255
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img=np.reshape(Xtr[i],(28,28))
    plt.imshow(img,cmap=plt.cm.binary)

'''
def ShowImg():
    X_train, y_train, X_test, y_test = load.Load()
    Xtr=X_train/255
    Xte=X_test/255
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img=np.reshape(Xtr[i],(28,28))
        plt.imshow(img,cmap=plt.cm.binary)

ShowImg()
'''
