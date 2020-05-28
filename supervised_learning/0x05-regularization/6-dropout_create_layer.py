#!/usr/bin/env python3
"""
Create a Layer with L2 Regularization
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout
    :param prev: tensor containing the output of the previous layer
    :param n: number of nodes the new layer should contain
    :param activation:  activation function that should be used on the layer
    :param keep_prob:  probability that a node will be kept
    :return: output of the new layer
    """
    k_i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    k_r = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=k_i, kernel_regularizer=k_r)
    return layer(prev)
