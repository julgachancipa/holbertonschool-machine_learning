#!/usr/bin/env python3
"""
Normalize
"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    """
    X -= m
    X /= s
    return X
