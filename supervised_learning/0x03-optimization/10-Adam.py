#!/usr/bin/env python3
"""
Adam Upgraded
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, epsilon=epsilon,
                                       beta1=beta1, beta2=beta2)
    train = optimizer.minimize(loss)
    return train
