#!/usr/bin/env python3
import tensorflow as tf
"""Layers"""


def create_layer(prev, n, activation):
    """Create Layer"""
    output = tf.layers.Dense(name='layer', units=n, activation=activation,
                             kernel_initializer=tf.contrib.layers.
                             variance_scaling_initializer(mode="FAN_AVG"))
    return output(prev)
