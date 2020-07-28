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

    def pdf(self, x):
        """
        calculates the PDF at a data point
        :param x: x is a numpy.ndarray of shape (d, 1) containing the data
        point whose PDF should be calculated
            d is the number of dimensions of the Multinomial instance
        :return: the value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')
        if x.shape[0] is not self.mean.shape[0]:
            raise ValueError('x must have the shape ({d}, 1)')
        d = x.shape[0]
        x_m = x - self.mean
        result = (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))) *
                  np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2))
        return float(result)
