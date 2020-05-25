#!/usr/bin/env python3
"""
Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    """
    sens = np.array([])

    for c in confusion:
        True_Pos = np.amax(c)
        actual = np.sum(c)

        sens = np.append(sens, (True_Pos / actual))

    return sens
