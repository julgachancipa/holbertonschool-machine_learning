#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network
    using gradient descent with L2 regularization
    :param Y: one-hot mtx that contains the correct labels for the data
    :param weights: dictionary of the weights and biases of the neural network
    :param cache: dictionary of the outputs of each layer of the neural network
    :param alpha: learning rate
    :param lambtha: L2 regularization parameter
    :param L: number of layers of the network
    :return: Nothing
    """
    m = (Y.shape[1])
    Al = cache["A" + str(L)]
    dAl = Al - Y
    for lay in reversed(range(1, L + 1)):
        Al = cache["A" + str(lay)]
        gl_d = 1 - np.power(Al, 2)
        if lay == L:
            dZl = dAl
        else:
            dZl = dAl * gl_d
        Al_1 = cache["A" + str(lay - 1)]
        dWl = (1 / m) * np.matmul(dZl, Al_1.T)
        dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)
        Wl = weights["W" + str(lay)]
        dAl = np.matmul(Wl.T, dZl)

        kW = "W" + str(lay)
        kb = "b" + str(lay)
        weights[kW] = weights[kW] - alpha * dWl
        weights[kb] = weights[kb] - alpha * dbl
