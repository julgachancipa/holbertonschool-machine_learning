#!/usr/bin/env python3
"""
PCA
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    :param X: is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    :param var:
    :return:
    """
    u, s, vh = np.linalg.svd(X)
    variance = np.cumsum(s) / np.sum(s)
    r = np.argwhere(variance >= var)[0, 0]
    return vh[:r + 1].T
