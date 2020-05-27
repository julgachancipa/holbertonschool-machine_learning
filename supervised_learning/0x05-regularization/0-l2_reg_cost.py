#!/usr/bin/env python3
"""
L2 Regularization Cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    :param cost: cost of the network without L2 regularization
    :param lambtha: regularization parameter
    :param weights: dictionary of the weights and biases
    :param L: number of layers in the neural network
    :param m:  number of data points used
    :return:  cost of the network accounting for L2 regularization
    """
    sum = 0
    for layer in range(1, L + 1):
        key = 'W' + str(layer)
        sum += np.linalg.norm(weights[key])
    return cost + ((lambtha/(2*m)) * sum)
