#!/usr/bin/env python3
"""
Q affinities
"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities
    :param Y: Y is a numpy.ndarray of shape (n, ndim) containing the low0
    dimensional transformation of X
        n is the number of points
        ndim is the new dimensional representation of X
    :return: Q, num
        Q is a numpy.ndarray of shape (n, n) containing the Q affinities
        num is a numpy.ndarray of shape (n, n) containing the numerator
        of the Q affinities
    """
    sum_Y = np.sum(np.square(Y), 1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0.)
    Q = num / np.sum(num)
    return Q, num
