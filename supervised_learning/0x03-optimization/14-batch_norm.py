#!/usr/bin/env python3
"""
Batch Normalization Upgraded
"""
import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural
    """
    lay = tf.layers.Dense(units=n,
                          kernel_initializer=tf.contrib.layers.
                          variance_scaling_initializer(mode="FAN_AVG"))
    z = lay(prev)

    mean, variance = tf.nn.moments(z, [0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    z_n = tf.nn.batch_normalization(z, mean, variance,
                                    beta, gamma, 1e-8)
    y_pred = activation(z_n)
    return y_pred
