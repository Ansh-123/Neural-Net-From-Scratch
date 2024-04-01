import numpy as np


# @jit(target_backend='cuda')
# def te(a):
#    for i in range(10000000):
#        a[i] += 1


# creates a dropout matrix to multiplied by array y to dropout a certain ratio of the numbers
def dropout(y, ratio):
    (m, n) = np.shape(y)
    ym = np.zeros((m*n,))

    idx = round(m*n*(1-ratio))
    high = 1 / (1-ratio)
    for i in range(idx):
        ym[i] = high
    return np.random.permutation(np.reshape(ym, (m, n)))


# relu function
def relu (x):
   y = np.multiply(np.maximum(0, x), .02)
   return y


# derivative of relu function
def drelu(v):
   return np.multiply((np.zeros(np.shape(v)) < v).astype(float), .02)


def softmax(v):
   ex = np.sum(np.exp(v))
   return np.divide(np.exp(v), ex)


# normalized softmax, to avoid overflow for user images
def nsoftmax(v):
    ex = np.sum(np.exp(v - np.max(v)))
    return np.divide(np.exp(v - np.max(v)), ex)


def dropoutRelu(W1, W2, W3, W4, m1, m2, m3, m4, X, D):
   alpha = 0.01
   beta = 0.90

   for i in range(len(X)):
       x = np.reshape(X[i], (784, 1))
       v1 = np.dot(W1, x)
       y1 = relu(v1)
       y1 = np.multiply(y1, dropout(y1, 0.2))

       v2 = np.dot(W2, y1)
       y2 = relu(v2)
       y2 = np.multiply(y2, dropout(y2, 0.2))

       v3 = np.dot(W3, y2)
       y3 = relu(v3)
       y3 = np.multiply(y3, dropout(y3, 0.2))

       v4 = np.dot(W4, y3)
       y4 = softmax(v4)

       e = np.subtract(np.array(D[i])[:, np.newaxis], y4)
       delta = e

       e3 = np.dot(np.transpose(W4), np.array(delta))
       delta3 = np.multiply(drelu(np.array(y3)), e3)

       e2 = np.dot(np.transpose(W3), delta3)
       delta2 = np.multiply(drelu(np.array(y2)), e2)

       e1 = np.dot(np.transpose(W2), delta2)
       delta1 = np.multiply(drelu(np.array(y1)), e1)

       DW4 = np.dot(np.array(np.multiply(alpha, delta)), y3.transpose())
       m4 = np.add(DW4, np.multiply(m4, beta))
       W4 = np.add(m4, W4)

       DW3 = np.dot(np.array(np.multiply(alpha, delta3)), y2.transpose())
       m3 = np.add(DW3, np.multiply(m3, beta))
       W3 = np.add(m3, W3)

       DW2 = np.dot(np.array(np.multiply(alpha, delta2)), y1.transpose())
       m2 = np.add(DW2, np.multiply(m2, beta))
       W2 = np.add(m2, W2)

       DW1 = np.dot(np.array(np.multiply(alpha, delta1)), np.transpose(x))
       m1 = np.add(DW1, np.multiply(m1, beta))
       W1 = np.add(m1, W1)

   return W1, W2, W3, W4, m1, m2, m3, m4
