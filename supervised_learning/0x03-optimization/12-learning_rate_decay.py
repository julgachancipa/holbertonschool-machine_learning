#!/usr/bin/env python3
"""
Learning Rate Decay Upgraded
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates learning rate operation
    """
    learning_rate = tf.train.inverse_time_decay(alpha, global_step,
                                                decay_step, decay_rate,
                                                staircase=True)
    return learning_rate
