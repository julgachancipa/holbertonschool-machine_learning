#!/usr/bin/env python3
"""
Variance
"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param C: is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    :return: var, or None on failure
        var is the total variance
    """
    if type(X) is np.ndarray:
        return None
    D = np.linalg.norm(X[:, None] - C, axis=-1)
    clss = np.argmin(D, axis=-1)
    return np.sum(np.sqrt(clss))
