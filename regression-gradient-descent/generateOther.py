import random

def h(x):
    return x[0] * 3 +x[1] * 5 - x[2] * 2

d = 3

input = []
output = []
for i in range(50):
    x = []
    for j in range(d):
        x.append((random.random() - 0.5) * 10)
    e = (random.random() - 0.5) * 5
    input.append(x)
    output.append(h(x) + e)

f = open("data.txt", "w")
f.write("input = " + str(input) + "\n\noutput = " + str(output))
f.close()