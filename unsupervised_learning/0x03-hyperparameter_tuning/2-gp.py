#!/usr/bin/env python3
"""
Update Gaussian Process
"""
import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, len=1, sigma_f=1):
        """
        Class constructor
        :param X_init: is a numpy.ndarray of shape (t, 1) representing the
        inputs already sampled with the black-box function
        :param Y_init: is a numpy.ndarray of shape (t, 1) representing the
        outputs of the black-box function for each input in X_init
            t is the number of initial samples
        :param l: is the length parameter for the kernel
        :param sigma_f: is the standard deviation given to the output of
        the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.len = len
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        :param X1: is a numpy.ndarray of shape (m, 1)
        :param X2: is a numpy.ndarray of shape (n, 1)
        :return:
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) +\
            np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points
        in a Gaussian process
        :param X_s: is a numpy.ndarray of shape (s, 1) containing all of the
        points whose mean and standard deviation should be calculated
            s is the number of sample points
        :return: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean for each
            point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the standard
            deviation for each point in X_s, respectively
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.T[0], np.diag(cov_s)

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process
        :param X_new: is a numpy.ndarray of shape (1,) that
        represents the new sample point
        :param Y_new: is a numpy.ndarray of shape (1,) that
        represents the new sample function value
        :return: nothing
        Updates the public instance attributes X, Y, and K
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
