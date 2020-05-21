#!/usr/bin/env python3
"""
Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural
    network using batch normalization
    """
    m = np.mean(Z, axis=0)
    s = np.std(Z, axis=0)
    Z_n = (Z - m) / (s + epsilon**(1/2))
    NZ = gamma * Z_n + beta

    return NZ
