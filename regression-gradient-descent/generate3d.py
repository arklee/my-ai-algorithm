import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

import random
import matplotlib.pyplot as plt
import numpy as np

# def h(x):
#     return np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2))

def h(x):
    return x[0]**2 - 2 * x[1]**2 + 6*x[0] - 6*x[1] + 3*x[0]*x[1] - 4

xs = []
zs = []
for i in range(100):
    x1 = (random.random() - 0.5) * 10
    x2 = (random.random() - 0.5) * 10
    e = (random.random() - 0.5) * 10
    xs.append([x1, x2])
    zs.append(h([x1, x2]) + e)

f = open("data.txt", "w")
f.write("input = " + str(xs) + "\n\noutput = " + str(zs))
f.close()

xs = np.array(xs)
zs = np.array(zs)

ax.scatter(xs[:, 0], xs[:, 1], zs, marker='v')

X = np.linspace(xs.min(axis=0)[0], xs.max(
    axis=0)[0], 256, endpoint=True)
Y = np.linspace(xs.min(axis=0)[1], xs.max(
    axis=0)[1], 256, endpoint=True)
X, Y = np.meshgrid(X, Y)
Z = h([X, Y])
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('Z')

plt.show()