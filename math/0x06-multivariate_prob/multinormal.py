#!/usr/bin/env python3
"""
Multivariate Normal distribution
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    :param X: numpy.ndarray of shape (n, d) containing the data set
    :return: mean, cov
    """
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n = X.shape[0]
    if n < 2:
        raise ValueError('X must contain multiple data points')
    mean = np.expand_dims(np.mean(X, axis=0), axis=0)
    X -= mean
    cov = np.dot(X.T, X) / (n - 1)
    return mean, cov


class MultiNormal ():
    """
    Multinormal Class
    """
    def __init__(self, data):
        """
        class constructor
        :param data: numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(data.shape) is not 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        n = data.shape[1]
        if n < 2:
            raise ValueError('data must contain multiple data points')
        mean, self.cov = mean_cov(data.T)
        self.mean = mean.T
