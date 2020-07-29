#!/usr/bin/env python3
"""
PCA v2
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    :param X: is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    :param ndim: is the new dimensionality of the transformed X
    :return:
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T
    return np.matmul(X_m, W)
