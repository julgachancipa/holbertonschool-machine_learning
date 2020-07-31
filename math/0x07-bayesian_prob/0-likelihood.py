#!/usr/bin/env python3
"""
Likelihood
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects
    :param x: number of patients that develop severe side effects
    :param n: is the total number of patients observed
    :param P: s a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    :return: a 1D numpy.ndarray containing the likelihood of obtaining
    the data, x and n, for each probability in P
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) is not 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    return (np.math.factorial(n) / (np.math.factorial(x) *
                                    np.math.factorial(n - x))) * (P ** x) *\
           ((1 - P) ** (n-x))
