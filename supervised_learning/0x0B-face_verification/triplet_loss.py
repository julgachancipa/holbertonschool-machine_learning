#!/usr/bin/env python3
"""
Initialize Triplet Loss
"""
import tensorflow as tf
import tensorflow.keras as K


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
        A, P, N = inputs

        positive_d = K.backend.sum(K.backend.square(A - P), axis=1)
        negative_d = K.backend.sum(K.backend.square(A - N), axis=1)
        loss = K.backend.maximum((positive_d - negative_d) + self.alpha, 0)
        return loss

    def call(self, inputs):
        """
        Adds the triplet loss to the graph
        :param inputs: list containing the anchor, positive,
        and negative output
        :return: the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
