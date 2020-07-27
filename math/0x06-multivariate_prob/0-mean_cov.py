#!/usr/bin/env python3
"""
Mean and Covariance
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    :param X: numpy.ndarray of shape (n, d) containing the data set
    :return: mean, cov
    """
    if type(X) is not np.ndarray and len(X.shape) is not 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n = X.shape[0]
    if n < 2:
        raise ValueError('X must contain multiple data points')
    mean = np.expand_dims(np.mean(X, axis=0), axis=0)
    X -= mean
    cov = np.dot(X.T, X) / (n - 1)
    return mean, cov
