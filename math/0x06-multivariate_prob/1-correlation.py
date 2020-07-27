#!/usr/bin/env python3
"""
Correlation
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    :param C: numpy.ndarray of shape (d, d) containing a covariance matrix
    :return: numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) is not 2 and C.shape[0] is not C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    D = np.sqrt(np.diag(C))
    outer = np.outer(D, D)
    return C / outer
