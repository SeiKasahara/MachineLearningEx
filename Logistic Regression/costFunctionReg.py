import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
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
    for j in range(theta.size):
        cost += lmd*(theta[j]**2)/(2*m)
        if j > 0:
            grad[j] += lmd*theta[j]/m
    # ===========================================================

    return cost, grad
