#!/usr/bin/env python3
"""
Layers
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Create Layer
    """
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=tf.contrib.layers.
                            variance_scaling_initializer(mode="FAN_AVG"))
    return layer(prev)
