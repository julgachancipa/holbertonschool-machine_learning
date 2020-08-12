#!/usr/bin/env python3
"""
Absorbing Chains
"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    :param P: P is a is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    :return: True if it is absorbing, or False on failure
    """
    if (P == np.eye(P.shape[0])).all():
        return True
    if np.any(np.diag(P) == 1):
        for i, row in enumerate(P):
            for j, col in enumerate(row):
                if i == j:
                    if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                        return False
        return True
    return False
