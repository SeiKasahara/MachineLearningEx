import numpy as np
from sigmoid import *


def lr_cost_function(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    #for i in range(m):
    #    cost += ((-y[i]) * np.log(sigmoid(np.dot(theta,X[i]))) - (1-y[i]) * np.log(1-sigmoid(np.dot(theta,X[i]))))/m
    #    grad += (sigmoid(np.dot(theta,X[i]))-y[i])*X[i]/m
    f = sigmoid(X @ theta.T)
    theta[0] = 0
    cost = ( (-y) @ (np.log(f)).T - ( (1-y) @ (np.log(1-f)).T ) )/m + (lmd / (2 * m))*(theta @ theta.T)
    grad = ((f - y) @ X)/m +lmd*theta/m
    # =========================================================

    return cost, grad
