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

    for i in range(confusion.shape[0]):
        TP = confusion[i][i]
        actual_yes = np.sum(confusion[i])

        sens = np.append(sens, (TP / actual_yes))

    return sens
