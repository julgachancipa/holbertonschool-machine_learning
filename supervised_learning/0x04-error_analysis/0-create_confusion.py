#!/usr/bin/env python3
"""
Create Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    """
    m = labels.shape[0]
    classes = labels.shape[1]

    conf_mtx = np.zeros((classes, classes))
    for ex in range(m):
        actual = np.argmax(labels[ex])
        predicted = np.argmax(logits[ex])

        conf_mtx[actual][predicted] += 1

    return conf_mtx
