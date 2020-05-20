#!/usr/bin/env python3
"""
Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    i = np.arange(X.shape[0])
    new_i = np.random.permutation(i)
    return X[new_i], Y[new_i]
