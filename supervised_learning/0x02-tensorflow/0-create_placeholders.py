#!/usr/bin/env python3
import tensorflow as tf
"""
Placeholders
"""


def create_placeholders(nx, classes):
    """
    returns two placeholders, x and y,
    for the neural network
    """
    x = tf.placeholder("float", [None, nx], name='x')
    y = tf.placeholder("float", [None, classes], name='y')

    return x, y
