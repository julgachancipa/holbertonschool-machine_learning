#!/usr/bin/env python3
"""
K-means
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    :param X: is a numpy.ndarray of shape (n, d) containing the dataset that
    will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    :param k: is a positive integer containing the number of clusters
    :return: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2\
            or type(k) is not int or k <= 0:
        return None
    _, d = X.shape
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    return np.random.uniform(low=low, high=high, size=(k, d))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    :param X: is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    :param k: is a positive integer containing the number of clusters
    :param iterations: is a positive integer containing the maximum number
    of iterations that should be performed
    :return: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to
    """
    if type(iterations) is not int:
        return None, None
    C = initialize(X, k)

    for _ in range(iterations):
        D = np.linalg.norm(X[:, None] - C, axis=-1)
        clss = np.argmin(D, axis=-1)
        for j in range(k):
            idx = np.argwhere(clss == j)
            if not len(idx):
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[idx], axis=0)
    return C, clss
