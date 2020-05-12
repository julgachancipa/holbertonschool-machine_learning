#!/usr/bin/env python3
"""One-Hot Decode"""
import numpy as np


def one_hot_decode(one_hot):
    """that converts a one-hot
    matrix into a vector of labels"""
    if one_hot is None or type(one_hot) != np.ndarray:
        return None
    for x in one_hot.flatten():
        if x != 0 or x != 1:
            return None
    return np.argmax(one_hot, axis=0)
