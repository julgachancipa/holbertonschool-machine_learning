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
        predicted = np.argmax(labels[ex])
        actual = np.argmax(logits[ex])

        conf_mtx[predicted][actual] += 1

    return conf_mtx
