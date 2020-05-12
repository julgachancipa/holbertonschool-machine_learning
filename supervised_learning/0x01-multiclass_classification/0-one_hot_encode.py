#!/usr/bin/env python3
"""One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """ that converts a numeric label
    vector into a one-hot matrix"""
    if classes < 3 or type(classes) is not int:
        return None
    if Y is None or type(Y) != np.ndarray:
        return None
    for c in Y:
        if c >= classes:
            return None
    m = Y.shape[0]
    mtx = np.zeros((m, classes))

    for row, c_label in zip(mtx, Y):
        row[c_label] = 1

    return mtx.T
