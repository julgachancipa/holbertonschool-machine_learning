#!/usr/bin/env python3
import tensorflow as tf
"""Accuracy"""


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    mean = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
    return mean
