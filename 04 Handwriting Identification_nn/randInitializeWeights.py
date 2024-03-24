import numpy as np


def rand_initialization(l_in, l_out):
    # You need to return the following variable correctly
    w = np.zeros((l_out, 1 + l_in))

    # ===================== Your Code Here =====================
    # Instructions : Initialize w randomly so that we break the symmetry while
    #                training the neural network
    #
    # Note : The first column of w corresponds to the parameters for the bias unit
    #
    ep = np.sqrt(6)/(np.sqrt(l_in+l_out))
    w = np.random.randint(l_out, 1+l_in, size=(l_out,l_in+1))*2*ep - ep

    # ===========================================================

    return w
