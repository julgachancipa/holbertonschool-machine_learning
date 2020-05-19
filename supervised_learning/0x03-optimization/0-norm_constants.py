#!/usr/bin/env python3
"""
Normalization Constants
"""


def normalization_constants(X):
    """
    Calculates the normalization (standardization)
    constants of a matrix
    """
    m = X.mean(0)
    s = X.std(0)
    return m, s
