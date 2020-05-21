#!/usr/bin/env python3
"""
Momentum Upgraded
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train = optimizer.minimize(loss)
    return train
