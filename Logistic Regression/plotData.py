import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #
    pos_data = X[y[:]==1]
    neg_data = X[y[:]!=1]
    plt.scatter(pos_data[:,0], pos_data[:,1], c='b', marker='+')
    plt.scatter(neg_data[:,0], neg_data[:,1], c='y', marker='o')
