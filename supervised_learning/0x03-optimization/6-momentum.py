#!/usr/bin/env python3
"""
Momentum Upgraded
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train = optimizer.minimize(loss)
    return train
