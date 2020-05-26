#!/usr/bin/env python3
"""
Specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    """
    spe = np.array([])
    sum_all = np.sum(confusion)

    for i in range(confusion.shape[0]):
        TP = confusion[i][i]
        TN = sum_all - np.sum(confusion[i]) - np.sum(confusion.T[i]) + TP
        FP = np.sum(confusion.T[i]) - TP
        actual_no = TN + FP

        spe = np.append(spe, (TN / actual_no))

    return spe
