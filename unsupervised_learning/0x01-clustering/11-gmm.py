#!/usr/bin/env python3
"""
GMM
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset
    :param X: is a numpy.ndarray of shape (n, d) containing the dataset
    :param k: is the number of clusters
    :return: pi, m, S, clss, bic
        pi is a numpy.ndarray of shape (k,) containing the cluster priors
        m is a numpy.ndarray of shape (k, d) containing the centroid means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices
        clss is a numpy.ndarray of shape (n,) containing the cluster indices
        for each data point
        bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
        value for each cluster size tested
    """
    mixture = sklearn.mixture.GaussianMixture(n_components=k)
    mixture_fit = mixture.fit(X)
    m = mixture_fit.means_
    S = mixture_fit.covariances_
    pi = mixture_fit.weights_
    clss = mixture.predict(X)
    bic = mixture.bic(X)

    return pi, m, S, clss, bic
