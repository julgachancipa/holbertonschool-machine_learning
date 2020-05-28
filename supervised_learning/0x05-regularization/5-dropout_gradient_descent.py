#!/usr/bin/env python3
"""
Gradient Descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout
    regularization using gradient descent
    :param Y:  one-hot mtx with the correct labels
    :param weights:
    :param cache:dictionary of the weights and biases of the neural network
    :param alpha: learning rate
    :param keep_prob: the probability that a node will be kept
    :param L: number of layers of the network
    :return: ntg
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
            dZl *= (cache["D" + str(lay)] / keep_prob)
        Wl = weights["W" + str(lay)]
        Al_1 = cache["A" + str(lay - 1)]
        dWl = (1 / m) * np.matmul(dZl, Al_1.T)
        dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)
        dAl = np.matmul(Wl.T, dZl)

        kW = "W" + str(lay)
        kb = "b" + str(lay)
        weights[kW] = weights[kW] - alpha * dWl
        weights[kb] = weights[kb] - alpha * dbl
