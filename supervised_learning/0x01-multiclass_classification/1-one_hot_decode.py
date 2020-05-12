#!/usr/bin/env python3
"""One-Hot Decode"""
import numpy as np


def one_hot_decode(one_hot):
    """that converts a one-hot
    matrix into a vector of labels"""

    classes = one_hot.shape[0]
    m = one_hot.shape[1]

    mtx = one_hot.T

    return np.argmax(mtx, axis=1)
