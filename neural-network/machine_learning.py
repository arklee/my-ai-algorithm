from mnist import MNIST
import numpy as np
from tqdm import tqdm
import datetime


def sigmoid(array):
    return 1/(1 + np.exp(-array))


# def sigmoidDiff(array):
#     return array * (np.ones(len(array)) - array)

# def relu(array):
#     def f(x):
#         return x if x > 0 else 0
#     return np.array(list(map(f, array)))

# def reluDiff(array):
#     def f(x):
#         if (x >= 0):
#             return 1
#         else:
#             return 0
#     return np.array(list(map(f, array)))

mndata = MNIST('samples')
images, labels = mndata.load_training()
images = np.array(images)
labels = np.array(labels)

th01 = np.random.randn(17, 784) * np.sqrt(1/17)
th12 = np.random.randn(17, 17) * np.sqrt(1/17)
th23 = np.random.randn(10, 17) * np.sqrt(1/10)

size = len(labels)

def propogation(input, output, batch, theta01, theta12, theta23):
    activator = sigmoid
    # activatorDiff = sigmoidDiff

    # cost = 0

    Delta01 = np.zeros((17, 784))
    Delta12 = np.zeros((17, 17))
    Delta23 = np.zeros((10, 17))

    for s in range(batch):

        # forword propogation

        a0 = input[s] / 255  # 784 x 1

        z1 = np.matmul(theta01, a0)
        a1 = activator(z1)  # 17 x 1
        a1[0] = 1

        z2 = np.matmul(theta12, a1)
        a2 = activator(z2)  # 17 x 1
        a2[0] = 1

        z3 = np.matmul(theta23, a2)
        a3 = activator(z3)  # 10 x 1

        # a1 = np.matmul(theta01, a0)
        # a1[a1 < 0] = 0
        # a1[0] = 1

        # a2 = np.matmul(theta12, a1)
        # a2[a2 < 0] = 0
        # a2[0] = 1

        # a3 = np.matmul(theta23, a2)
        # a3[a3 < 0] = 0

        y = np.zeros(10)
        y[output[s]] = 1

        # for i in range(10):
        #     # cost += (y[i] - 1) * np.log(a3[i])
        #     if i == output[s]:
        #         cost += np.log(a3[i])
        #     else:
        #         cost += np.log(1 - a3[i])

        # backword propogation

        delta3 = a3 - y # 10 x 1
        delta2 = np.matmul(theta23.T, delta3) * (a2 * (1 - a2))  # activatorDiff(a2) # 17 x 1
        delta1 = np.matmul(theta12.T, delta2) * (a1 * (1 - a1))  # activatorDiff(a1) # 17 x 1

        Delta01 += np.matmul(delta1[np.newaxis].T, a0[np.newaxis])  # 17 x 784
        Delta12 += np.matmul(delta2[np.newaxis].T, a1[np.newaxis])  # 17 x 17
        Delta23 += np.matmul(delta3[np.newaxis].T, a2[np.newaxis])  # 10 x 17

    lmd = 1
    t01 = theta01.copy()
    t12 = theta12.copy()
    t23 = theta23.copy()
    t01[:, 0] = 0
    t12[:, 0] = 0
    t23[:, 0] = 0
    Delta01 += t01 * lmd
    Delta12 += t12 * lmd
    Delta23 += t23 * lmd

    # cost = -cost / size

    # print("代价为：" + str(cost))

    return (Delta01, Delta12, Delta23)


start = datetime.datetime.now()

alpha = 1
n = 10000
batch = 128

for i in tqdm(range(n)):
    sample = np.random.randint(0,60000,size=batch)
    # sample = range(batch)
    # alpha = math.exp(-i / 3)
    d = propogation(images[sample], labels[sample], batch, th01, th12, th23)
    th01 = th01 - alpha / batch * d[0]
    th12 = th12 - alpha / batch * d[1]
    th23 = th23 - alpha / batch * d[2]

end = datetime.datetime.now()

print("共耗时：" + str(end - start))

f = open("data.py", "w")
f.write("theta = [0, 0, 0]\n\ntheta[0] = " + str(th01.tolist()) + "\n\ntheta[1] = " + str(
    th12.tolist()) + "\n\ntheta[2] = " + str(th23.tolist()))
f.close()
