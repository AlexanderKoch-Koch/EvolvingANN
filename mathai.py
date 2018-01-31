import math
import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return 1 / (1 + math.e ** -x) * (1 - 1 / (1 + math.e ** -x))
    else:
        return 1 / (1 + math.e ** -x)


def relu(X, derivative=False):
    if derivative:
        X[X <= 0] = 0
        X[X > 0] = 1
    else:
        np.maximum(X, 0, out=X)

    return X

