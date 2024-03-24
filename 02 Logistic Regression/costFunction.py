import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    for i in range(m):
        cost += ((-y[i]) * np.log(sigmoid(np.dot(theta,X[i]))) - (1-y[i]) * np.log(1-sigmoid(np.dot(theta,X[i]))))/m
        grad += (sigmoid(np.dot(theta,X[i]))-y[i])*X[i]/m
    # ===========================================================
    print(cost, grad)
    return cost, grad
