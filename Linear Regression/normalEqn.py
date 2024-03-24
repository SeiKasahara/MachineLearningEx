import numpy as np

def normal_eqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    # ===================== Your Code Here =====================
    # Instructions : Complete the code to compute the closed form solution
    #                to linear regression and put the result in theta
    #

    theta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    return theta
