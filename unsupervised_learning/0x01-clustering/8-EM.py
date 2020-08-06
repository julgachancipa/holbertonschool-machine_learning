#!/usr/bin/env python3
"""
EM: expectation maximization
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    :param X: is a numpy.ndarray of shape (n, d) containing the data set
    :param k: is a positive integer containing the number of clusters
    :param iterations: is a positive integer containing the maximum number
    of iterations for the algorithm
    :param tol: is a non-negative float containing tolerance of the log
    likelihood, used to determine early stopping
    :param verbose: is a boolean that determines if you should print
    information about the algorithm
    :return: pi, m, S, g, l, or None, None, None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors for each
        cluster
        m is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster
        g is a numpy.ndarray of shape (k, n) containing the probabilities
        for each data point in each cluster
        l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None, None, None)
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return (None, None, None, None, None)
    if type(iterations) != int or iterations <= 0:
        return (None, None, None, None, None)
    if type(tol) != float or tol < 0:
        return (None, None, None, None, None)
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    prev_lkhood = 0

    for i in range(iterations):
        g, lkhood = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if verbose:
            message = 'Log Likelihood after {} iterations: {}'\
                .format(i, lkhood.round(5))
            if (i % 10 == 0) or (i == 0):
                print(message)
            if i == (iterations - 1):
                print(message)

            if abs(lkhood - prev_lkhood) <= tol:
                print(message)
                break
            if i == (iterations + 1):
                break
        if abs(lkhood - prev_lkhood) <= tol:
            break
        prev_lkhood = lkhood
    return pi, m, S, g, lkhood
