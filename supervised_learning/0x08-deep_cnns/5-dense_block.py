#!/usr/bin/env python3
"""
Dense Block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected
    Convolutional Networks
    :param X: is the output from the previous layer
    :param nb_filters: is an integer representing the number of filters in X
    :param growth_rate: is the growth rate for the dense block
    :param layers: is the number of layers in the dense block
    :return: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """
    concat = X
    for _ in range(layers):
        batch_normalization_a = K.layers.BatchNormalization()(concat)
        activation_a = K.layers.Activation('relu')(batch_normalization_a)
        conv2d_a = K.layers.Conv2D(
            nb_filters * 2,
            1,
            padding='same',
            kernel_initializer='he_normal')(activation_a)
        batch_normalization_b = K.layers.BatchNormalization()(conv2d_a)
        activation_b = K.layers.Activation('relu')(batch_normalization_b)
        conv2d_b = K.layers.Conv2D(
            growth_rate,
            3,
            padding='same',
            kernel_initializer='he_normal')(activation_b)
        concat = K.layers.concatenate([concat, conv2d_b])

    return concat, nb_filters + growth_rate * layers
