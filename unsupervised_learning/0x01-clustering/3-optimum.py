#!/usr/bin/env python3
"""
Optimize k
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param kmin: is a positive integer containing the minimum number of
    clusters to check for (inclusive)
    :param kmax: is a positive integer containing the maximum number of
    clusters to check for (inclusive)
    :param iterations: is a positive integer containing the maximum number
    of iterations for K-means
    :return: results, d_vars, or None, None on failure
        results is a list containing the outputs of K-means for each
        cluster size
        d_vars is a list containing the difference in variance from the
        smallest cluster size for each cluster size
    """
    if kmax:
        if kmin >= kmax:
            return None, None
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
            return None, None
        if type(kmin) != int or kmin <= 0 or kmax >= X.shape[0]:
            return None, None
        if type(iterations) != int or iterations <= 0:
            return None, None
    else:
        kmax = 1
    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results += [C, clss]
        var = variance(X, C)
        if k == kmin:
            min_var = var
        d_vars += [min_var - var]
    return results, d_vars
