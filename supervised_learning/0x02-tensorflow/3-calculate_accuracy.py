#!/usr/bin/env python3
"""
Accuracy
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    """
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
    return accuracy
