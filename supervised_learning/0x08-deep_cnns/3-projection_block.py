#!/usr/bin/env python3
"""
Projection Block
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    builds a projection block as described in Deep Residual Learning for Image
    Recognition (2015)
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution as well
        as the 1x1 convolution in the shortcut connection
    :param s: is the stride of the first convolution in both the main path
    and the shortcut connection
    :return: the activated output of the projection block
    """
    F11, F3, F12 = filters

    conv2d = K.layers.Conv2D(F11, 1, padding='same',
                             activation='relu', strides=s,
                             kernel_initializer='he_normal')(A_prev)
    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activation = K.layers.Activation('relu')(batch_normalization)

    conv2d_1 = K.layers.Conv2D(F3, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(activation)
    batch_normalization_1 = K.layers.BatchNormalization()(conv2d_1)
    activation_1 = K.layers.Activation('relu')(batch_normalization_1)

    conv2d_2 = K.layers.Conv2D(F12, 1, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(activation_1)
    batch_normalization_2 = K.layers.BatchNormalization()(conv2d_2)

    conv2d_3 = K.layers.Conv2D(F12, 1, padding='same',
                               activation='relu', strides=s,
                               kernel_initializer='he_normal')(A_prev)
    batch_normalization_3 = K.layers.BatchNormalization()(conv2d_3)

    add = K.layers.Add()([batch_normalization_2, batch_normalization_3])
    activation_2 = K.layers.Activation('relu')(add)
    return activation_2