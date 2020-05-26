#!/usr/bin/env python3
"""
F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix
    """
    s = np.power(sensitivity(confusion), -1)
    p = np.power(precision(confusion), -1)
    return 2 / (s + p)
