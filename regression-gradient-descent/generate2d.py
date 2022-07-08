import random
import matplotlib.pyplot as plt
import numpy as np

def h(x):
    return x[0] ** 2  - 8 * x[0] - 5

input = []
output = []
for i in range(50):
    x = (random.random()) * 10
    e = (random.random() - 0.5) * 4
    input.append([x])
    output.append(h([x]) + e)
    
f = open("data.txt", "w")
f.write("input = " + str(input) + "\n\noutput = " + str(output))
f.close()

input = np.array(input)
output = np.array(output)

plt.plot(input[:, 0], output, 'ro')

X = np.linspace(input.min(axis=0)[0], input.max(
                axis=0)[0], 256, endpoint=True)
Y = h([X])
plt.plot(X, Y)

plt.ylabel('z axis')
plt.xlabel('x axis')
plt.show()