import main as m
import numpy as np
import idx2numpy
import time
from numpy import loadtxt


print("loading data")
startTime = time.time()
# Read training and testing images and labels
file = 'samples/train-images-idx3-ubyte'
subTrainImages = idx2numpy.convert_from_file(file)
file = 'samples/t10k-images-idx3-ubyte'
subTestImages = idx2numpy.convert_from_file(file)
file = 'samples/t10k-labels-idx1-ubyte'
subTestLabels = idx2numpy.convert_from_file(file)
file = 'samples/train-labels-idx1-ubyte'
subTrainLabels = idx2numpy.convert_from_file(file)
trainLabels = []
testLabels = []

# load data for model
W1 = loadtxt('W1.csv', delimiter=',')
W2 = loadtxt('W2.csv', delimiter=',')
W3 = loadtxt('W3.csv', delimiter=',')
W4 = loadtxt('W4.csv', delimiter=',')

# Convert training labels to array of 10 based on value in the image
for i in range(len(subTrainLabels)):
    if subTrainLabels[i] == 0:
        trainLabels.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif subTrainLabels[i] == 1:
        trainLabels.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif subTrainLabels[i] == 2:
        trainLabels.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif subTrainLabels[i] == 3:
        trainLabels.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif subTrainLabels[i] == 4:
        trainLabels.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif subTrainLabels[i] == 5:
        trainLabels.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif subTrainLabels[i] == 6:
        trainLabels.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif subTrainLabels[i] == 7:
        trainLabels.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif subTrainLabels[i] == 8:
        trainLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif subTrainLabels[i] == 9:
        trainLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

# Convert testing tables to the same encoding.
for i in range(len(subTestLabels)):
    if subTestLabels[i] == 0:
        testLabels.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif subTestLabels[i] == 1:
        testLabels.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif subTestLabels[i] == 2:
        testLabels.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif subTestLabels[i] == 3:
        testLabels.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif subTestLabels[i] == 4:
        testLabels.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif subTestLabels[i] == 5:
        testLabels.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif subTestLabels[i] == 6:
        testLabels.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif subTestLabels[i] == 7:
        testLabels.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif subTestLabels[i] == 8:
        testLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif subTestLabels[i] == 9:
        testLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


trainImages = np.divide(subTrainImages[0:60000], 255)
testImages = np.divide(subTestImages[0:10000], 255)
loss = 0
correct = 0
print("running model")
for j in range(len(testImages)):
    x = np.reshape(testImages[j], (784, 1))

    v1 = np.dot(W1, x)
    y1 = m.relu(v1)

    v2 = np.dot(W2, y1)
    y2 = m.relu(v2)

    v3 = np.dot(W3, y2)
    y3 = m.relu(v3)

    v = np.dot(W4, y3)
    y = m.softmax(v)

    if np.argmax(y) == np.argmax(testLabels[j]):
        loss = loss + 1 - y[np.argmax(testLabels[j])]
        correct = correct + 1
    else:
        # print(np.transpose(y))
        # print(np.array(trainLabels[j])[np.newaxis])
        loss = loss + 1 - y[np.argmax(testLabels[j])]

endTime = time.time()
print("num correct")
print(correct)
print("loss")
print(loss / len(testImages))
print(endTime-startTime)





