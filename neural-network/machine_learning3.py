from mnist import MNIST
import numpy as np
from tqdm import tqdm
import datetime

print("正在读取数据......")
mndata = MNIST('samples')
images, labels = mndata.load_training()

labelsArray = np.zeros((60000, 10)) # 60000 x 10
for i in range(60000):
    labelsArray[i][labels[i]] = 1
# labelsArray[range(labelsArray.shape[0]),y] = 1
    
imagesArray = np.array(images)
    
def init(x,y):
    layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
    return layer.astype(np.float32)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

alpha = 0.01
epochs = 10000
batch = 128

theta01 = init(784, 16) # 784 x 16
theta12 = init(16, 16) # 16 x 16
theta23 = init(16, 10) # 16 x 10

start = datetime.datetime.now()

for i in tqdm(range(epochs)):
    # cost = 0
    
    sample = np.random.randint(0,60000,size=batch)

    # forword propogation
    a0 = imagesArray[sample] / 255 # 128 x 784
    z1 = a0 @ theta01 # 128 x 16
    a1 = sigmoid(z1)
    z2 = a1 @ theta12 # 128 x 16
    a2 = sigmoid(z2)
    z3 = a2 @ theta23 # 128 x 10
    a3 = sigmoid(z3)
    
    # z1 = a0 @ theta01 # 128 x 16
    # a1 = sigmoid(z1)
    # z2 = a1 @ theta12 # 128 x 16
    # a2 = sigmoid(z2)
    # z3 = a2 @ theta23 # 128 x 10
    # a3 = sigmoid(z3)
    
    # for i in range(10):
    #     if i == labels[s]:
    #         cost += np.log(a3[i])
    #     else:
    #         cost += np.log(1 - a3[i])
    
    # backword propogation
    delta3 = a3 - labelsArray[sample] # 128 x 10
    delta2 = delta3 @ theta23.T * (a2 * (1 - a2)) # 128 x 16
    delta1 = delta2 @ theta12.T * (a1 * (1 - a1)) # 128 x 16
    
    # cost = -cost / size
    # print("代价为：" + str(cost))
    
    theta01 = theta01 - alpha * (a0.T @ delta1)
    theta12 = theta12 - alpha * (a1.T @ delta2)
    theta23 = theta23 - alpha * (a2.T @ delta3)
    
end = datetime.datetime.now()

print("共耗时：" + str(end - start))

f = open("data3.py", "w")
f.write("theta = [0, 0, 0]\n\ntheta[0] = " + str(theta01.tolist()) + "\n\ntheta[1] = "
        + str(theta12.tolist()) + "\n\ntheta[2] = " + str(theta23.tolist()))
f.close()
