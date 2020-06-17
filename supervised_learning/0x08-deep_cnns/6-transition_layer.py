#!/usr/bin/env python3
"""
Transition Layer
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    that builds a transition layer as described in Densely Connected
    Convolutional Networks
    :param X: is the output from the previous layer
    :param nb_filters: is an integer representing the number of filters in X
    :param compression: is the compression factor for the transition layer
    :return: The output of the transition layer and the number of filters
    within the output, respectively
    """
    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv2d = K.layers.Conv2D(int(nb_filters * compression), 1, padding='same',
                             kernel_initializer='he_normal')(activation)
    average_pooling2d = K.layers.AveragePooling2D(2, strides=2,
                                                  padding='same')(conv2d)
    return average_pooling2d, average_pooling2d.shape[-1]
