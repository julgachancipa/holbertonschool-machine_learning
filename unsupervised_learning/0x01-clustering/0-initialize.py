#!/usr/bin/env python3
"""
Initialize K-means
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
    _, d = X.shape
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    return np.random.uniform(low=low, high=high, size=(k, d))
