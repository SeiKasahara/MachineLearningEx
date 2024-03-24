import numpy as np
from sigmoid import *
from sigmoidgradient import *

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):
    # Reshape nn_params back into the parameters theta1 and theta2, the weight 2-D arrays
    # for our two layer neural network
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

    # Useful value
    m = y.size

    # You need to return the following variables correctly
    cost = 0
    theta1_grad = np.zeros(theta1.shape)  # 25 x 401
    theta2_grad = np.zeros(theta2.shape)  # 10 x 26

    # ===================== Your Code Here =====================
    # Instructions : You should complete the code by working thru the
    #                following parts
    #
    # Part 1 : Feedforward the neural network and return the cost in the
    #          variable cost. After implementing Part 1, you can verify that your
    #          cost function computation is correct by running ex4.py
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         theta1_grad and theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to theta1 and theta2 in theta1_grad and
    #         theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the 
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to theta1_grad
    #               and theta2_grad from Part 2.
    #
    X = np.insert(X, 0, 1, axis=1) #bias a_1 = x
    layer_1_output = sigmoid(X @ theta1.T) #a_2 = g(z_1= x @ theta1.T)
    layer_1_output = np.insert(layer_1_output, 0, 1, axis=1) #bias
    h = sigmoid(layer_1_output @ theta2.T) #a_3 = g(z_2= a_2 @ theta2.T)
    theta1_copy = theta1.copy()
    theta2_copy = theta2.copy()
    theta1_copy[:,0] = 0
    theta2_copy[:,0] = 0
    theta1_reg = np.sum(np.power(theta1_copy,2))
    theta2_reg = np.sum(np.power(theta2_copy,2))
    for k in range(1, num_labels+1):
        y_k = (y == k).astype(int)
        first_log_k = (-y_k) @ np.log(h.T[k-1])
        second_log_k = (1-y_k) @ (np.log(1-h.T[k-1]))
        cost += (first_log_k - second_log_k)/m
    cost += lmd*(theta1_reg + theta2_reg)/(2*m)

    error_output = np.zeros(h.shape)
    
    # Gradient Descent & Backpropagation
    for k in range(1, num_labels+1):
        y_k = (y == k).astype(int)
        error_output[:,k-1] = h[:,k-1] - y_k #a_3 - y
    grad_2 = (error_output.T @ layer_1_output)/m
    theta2_grad += grad_2 + lmd*theta2_copy/m
    error_hidden_layer = (error_output @ theta2)[:,1:] * sigmoid_gradient(X @ theta1.T)
    theta1_grad += (error_hidden_layer.T @ X + lmd*theta1_copy)/m
    # ====================================================================================
    # Unroll gradients
    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()])

    return cost, grad
