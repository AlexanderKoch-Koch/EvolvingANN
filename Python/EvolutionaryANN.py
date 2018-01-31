from mathai import sigmoid
from mathai import relu
import numpy as np

eta = 0.1
x = np.array([0.2, 0.8])
y = np.array([1, 0.2, 0.2])

W1 = np.random.random((2, 3))
W2 = np.random.random((3, 2))




for episode in range(9):
    s1 = np.dot(x, W1)
    l1 = relu(s1)

    s2 = np.dot(l1, W2)
    l2 = relu(s2)
    e = np.square(np.subtract(l1, y))

    print(e)

    dW2 = np.multiply(np.subtract(l2, y) * relu(s2, derivative=True), l2)
    np.add(W2, eta * -dW2, out=W2)

    dW = np.multiply(dW2 * relu(s1, derivative=True), W1)
    np.add(W1, eta * -dW, out=W1)

    if episode % 10 == 0:
        shape = W1.shape
        sum_of_weights = np.sum(W1, axis=0)
        indeces = np.where(sum_of_weights < 0.4)
        # delete useless neurons
        W1 = np.delete(W1, indeces, axis=1)
        print(sum_of_weights)

    if episode % 30 == 0:
        # add neuron
        W1_new = np.random.random((2, 1))
        W1 = np.column_stack([W1, W1_new])


