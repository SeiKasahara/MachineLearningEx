import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    

    # ==========================================================
    for i in range(m):
        cost += 1/(2*m)*np.dot((np.dot(X,theta)-y),(np.dot(X,theta)-y))

    return cost
