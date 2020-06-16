#!/usr/bin/env python3
"""
Inception Block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    Going Deeper with Convolutions (2014)
    :param A_prev: output from the previous layer
    :param filters:  tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution
        before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution
        before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution
        after the max pooling
    :return: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv2d = K.layers.Conv2D(F1, 1,
                             activation='relu',
                             kernel_initializer='he_normal')(A_prev)

    conv2d_1 = K.layers.Conv2D(F3R, 1,
                               activation='relu',
                               kernel_initializer='he_normal')(A_prev)
    conv2d_2 = K.layers.Conv2D(F3, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(conv2d_1)

    conv2d_3 = K.layers.Conv2D(F5R, 1,
                               activation='relu',
                               kernel_initializer='he_normal')(A_prev)
    conv2d_4 = K.layers.Conv2D(F5, 5, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(conv2d_3)

    max_pooling2d = K.layers.MaxPool2D(3, 1, padding='same')(A_prev)
    conv2d_5 = K.layers.Conv2D(FPP, 1,
                               activation='relu',
                               kernel_initializer='he_normal')(max_pooling2d)

    inception_block = K.layers.concatenate(
        [conv2d, conv2d_2, conv2d_4, conv2d_5])
    return inception_block
