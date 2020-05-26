#!/usr/bin/env python3
"""
Precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    """
    pre = np.array([])

    for i in range(confusion.shape[0]):
        TP = confusion[i][i]
        predicted_yes = np.sum(confusion.T[i])

        pre = np.append(pre, (TP / predicted_yes))

    return pre
