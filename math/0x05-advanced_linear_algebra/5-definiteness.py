#!/usr/bin/env python3
"""
Definiteness
"""
import numpy as np


def definiteness(matrix):
    """
    calculates the definiteness of a matrix
    :param matrix: numpy.ndarray of shape (n, n) whose definiteness
    should be calculated
    :return: the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite
    """
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) != 2:
        return None
    n, m = matrix.shape
    if n is not m:
        return None
    w, v = np.linalg.eig(matrix)

    pos, neg, zer = 0, 0, 0
    for e_val in w:
        if e_val > 0:
            pos += 1
        elif e_val < 0:
            neg += 1
        else:
            zer += 1
    if pos and not neg and not zer:
        return 'Positive definite'
    elif pos and not neg and zer:
        return 'Positive semi-definite'
    elif not pos and neg and not zer:
        return 'Negative definite'
    elif not pos and neg and zer:
        return 'Negative semi-definite'
    return 'Indefinite'
