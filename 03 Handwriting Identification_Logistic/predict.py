import numpy as np
from sigmoid import *

def predict(theta1, theta2, x):
    # Useful values
    m = x.shape[0]
    num_labels = theta2.shape[0]

    # You need to return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned neural network. You should set p to a
    #                1-D array containing labels between 1 to num_labels.
    #
    x = np.insert(x, 0, 1, axis=1)
    layer_1_output = sigmoid(x @ theta1.T)
    layer_1_output = np.insert(layer_1_output, 0, 1, axis=1)
    layer_2_output = sigmoid(layer_1_output @ theta2.T)
    p = np.argmax(layer_2_output, axis=1)
    p += 1
    return p


