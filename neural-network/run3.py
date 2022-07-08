from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from data3 import theta

mndata = MNIST('samples')


def softmax(x):
    exponents = np.exp(x)
    return exponents / np.sum(exponents)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run(input):
    a0 = input / 255
    z1 = a0 @ theta[0]
    a1 = sigmoid(z1)
    z2 = a1 @ theta[1]
    a2 = sigmoid(z2)
    z3 = a2 @ theta[2]
    a3 = softmax(z3)

    return a3


images2, labels2 = mndata.load_testing()

# x=2341
# img = np.array(images2[x])
img = mpimg.imread('test.png')[:, :, 1].reshape(784, ) * 255

res = run(img)
print("结果：")
for i in range(10):
    print(str(i) + " 的可能性：" + "{:.8%}".format(res[i]))
print("预测：" + str(np.argmax(res)))
set1 = np.array(img)
plt.imshow(set1.reshape((28, 28)) / 255, plt.cm.gray)
# print("实际：" + str(labels2[x]))
plt.show()


