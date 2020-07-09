#!/usr/bin/env python3
"""
Initialize Triplet Loss
"""
import tensorflow as tf


class TripletLoss(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        """
        Class constructor
        :param alpha: alpha value used to calculate the triplet loss
        :param kwargs: extra parameters
        """
        self.alpha = alpha
        tf.keras.layers.Layer.__init__(self, **kwargs)

    def triplet_loss(self, inputs):
        """
        :param inputs: list containing the anchor, positive and negative
        output tensors from the last layer of the model, respectively
        :return: tensor containing the triplet loss values
        """
