#!/usr/bin/env python3
"""One-Hot Decode"""
import numpy as np


def one_hot_decode(one_hot):
    """that converts a one-hot
    matrix into a vector of labels"""
    for col in one_hot.T:
        if np.sum(col) != 1:
            return None
    return np.argmax(one_hot, axis=0)
