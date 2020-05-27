#!/usr/bin/env python3
"""
Create a Layer with L2 Regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tensorflow layer that includes L2 regularization
    :param prev: tensor containing the output of the previous layer
    :param n: number of nodes the new layer should contain
    :param activation: activation function that should be used on the layer
    :param lambtha: L2 regularization parameter
    :return: output of the new layer
    """
    k_i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    k_r = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=k_i, kernel_regularizer=k_r)
    return layer(prev)
