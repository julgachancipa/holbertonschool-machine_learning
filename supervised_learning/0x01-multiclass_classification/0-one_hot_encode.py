#!/usr/bin/env python3
"""One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """ that converts a numeric label
    vector into a one-hot matrix"""
    if classes < 3 or type(classes) is not int:
        return None
    if Y is None:
        return None
    m = Y.shape[0]
    mtx = np.zeros((m, classes))

    for col, c_label in zip(mtx, Y):
        col[c_label] = 1

    return mtx.T
