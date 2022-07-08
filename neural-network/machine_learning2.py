from mnist import MNIST
import numpy as np
from tqdm import tqdm
import datetime

def init(x,y):
    layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
    return layer.astype(np.float32)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# def softmax(x):
#     exponents=np.exp(x)
#     return exponents/np.sum(exponents)

print("......正在读取数据......")
mndata = MNIST('samples')
images, labels = mndata.load_training()

labelsArray = np.zeros((60000, 10)) # 60000 x 10
labelsArray[range(60000), labels] = 1
    
imagesArray = np.array(images)

alpha = 0.01
epochs = 10000
batch = 128

theta01 = init(784, 16) # 784 x 16
theta12 = init(16, 16) # 16 x 16
theta23 = init(16, 10) # 16 x 10

b1 = np.zeros(16) # 784 x 16
b2 = np.zeros(16) # 16 x 16
b3 = np.zeros(10) # 16 x 10

start = datetime.datetime.now()

for i in tqdm(range(epochs)):   
    sample = np.random.randint(0,60000,size=batch)

    # forword propogation
    a0 = imagesArray[sample] / 255 # 128 x 784
    z1 = a0 @ theta01 + b1[np.newaxis] # 128 x 16
    a1 = sigmoid(z1)
    z2 = a1 @ theta12 + b2[np.newaxis] # 128 x 16
    a2 = sigmoid(z2)
    z3 = a2 @ theta23 + b3[np.newaxis] # 128 x 10
    a3 = sigmoid(z3)
    
    # backword propogation
    delta3 = a3 - labelsArray[sample] # 128 x 10
    delta2 = delta3 @ theta23.T * (a2 * (1 - a2)) # 128 x 16
    delta1 = delta2 @ theta12.T * (a1 * (1 - a1)) # 128 x 16
    
    theta01 = theta01 - alpha * (a0.T @ delta1)
    theta12 = theta12 - alpha * (a1.T @ delta2)
    theta23 = theta23 - alpha * (a2.T @ delta3)
    
    b1 = b1 - alpha * np.sum(delta1, axis=0)
    b2 = b2 - alpha * np.sum(delta2, axis=0)
    b3 = b3 - alpha * np.sum(delta3, axis=0)
    
end = datetime.datetime.now()

print("共耗时：" + str(end - start))

f = open("data2.py", "w")
f.write("theta = [0, 0, 0]\n\ntheta[0] = " + str(theta01.tolist()) + "\n\ntheta[1] = "
        + str(theta12.tolist()) + "\n\ntheta[2] = " + str(theta23.tolist()) +"\n\nb = [0, 0, 0]\n\nb[0] = "
        + str(b1.tolist()) + "\n\nb[1] = " + str(b2.tolist()) + "\n\nb[2] = " + str(b3.tolist()))
f.close()
