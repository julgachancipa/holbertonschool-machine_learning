#!/usr/bin/env python3
"""
RMSProp Upgraded
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, epsilon=epsilon,
                                          decay=beta2)
    train = optimizer.minimize(loss)
    return train
