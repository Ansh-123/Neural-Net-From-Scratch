from numpy import loadtxt
import numpy as np
from PIL import Image
import main as m

W1 = loadtxt('W1.csv', delimiter=',')
W2 = loadtxt('W2.csv', delimiter=',')
W3 = loadtxt('W3.csv', delimiter=',')
W4 = loadtxt('W4.csv', delimiter=',')

image = Image.open('twote.png')
data = np.asarray(image)

fi = []
for i in data:
    temp1 = []
    for j in i:
        temp1.append(j[0])
    fi.append(temp1)

print(np.shape(fi))

x = np.reshape(fi, (784, 1))

v1 = np.dot(W1, x)
y1 = m.relu(v1)

v2 = np.dot(W2, y1)
y2 = m.relu(v2)

v3 = np.dot(W3, y2)
y3 = m.relu(v3)

v = np.dot(W4, y3)
y = m.nsoftmax(v)

print(y)

print(np.argmax(y))
