#!/usr/bin/env python3
"""
PDF
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    :param X: is a numpy.ndarray of shape (n, d) containing the data points
    whose PDF should be evaluated
    :param m: is a numpy.ndarray of shape (d,) containing the mean of the
    distribution
    :param S: is a numpy.ndarray of shape (d, d) containing the covariance
    of the distribution
    :return: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for
        each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray):
        return None
    if not isinstance(S, np.ndarray) or len(X.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    d = m.shape[0]
    x_m = X - m
    S_inv = np.linalg.inv(S)
    fac = np.einsum('...k,kl,...l->...', x_m, S_inv, x_m)
    P = (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(S)))) * \
        np.exp(-fac / 2)
    return np.maximum(P, 1e-300)
