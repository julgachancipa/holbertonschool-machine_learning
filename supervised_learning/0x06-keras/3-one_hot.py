#!/usr/bin/env python3
"""One hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix
    :param labels: vector
    :param classes: number of classes
    :return: one-hot matrix
    """
    return K.utils.to_categorical(labels, classes)
