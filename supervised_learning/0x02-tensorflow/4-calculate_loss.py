#!/usr/bin/env python3
import tensorflow as tf
"""
Loss
"""


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
