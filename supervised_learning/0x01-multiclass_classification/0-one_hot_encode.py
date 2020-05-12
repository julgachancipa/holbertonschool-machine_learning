#!/usr/bin/env python3
"""One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """ that converts a numeric label
    vector into a one-hot matrix"""
    m = Y.shape[0]

    if all(i >= classes for i in Y) or not m or type(classes) != int or classes < 2:
        return None
    mtx = np.zeros((m, classes))

    for col, c_label in zip(mtx, Y):
        col[c_label] = 1

    return mtx.T
