#!/usr/bin/env python3
"""
Accuracy
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    """
    accuracy = tf.reduce_mean((y, y_pred))
    return accuracy
