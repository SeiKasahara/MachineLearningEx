import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]

    # Return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned logistic regression parameters.
    #                You should set p to a 1D-array of 0's and 1's
    #
    for i in range(m):
        p_eval = sigmoid(np.dot(theta,X[i]))
        if p_eval > 0.5:
            p[i] = 1
        else:
            p[i] = 0

    # ===========================================================

    return p
