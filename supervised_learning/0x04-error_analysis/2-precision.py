#!/usr/bin/env python3
"""
Precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    """
    sens = np.array([])

    for c in confusion.T:
        True_Pos = np.amax(c)
        predicted = np.sum(c)

        sens = np.append(sens, (True_Pos / predicted))

    return sens
