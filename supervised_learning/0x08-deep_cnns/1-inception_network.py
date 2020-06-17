#!/usr/bin/env python3
"""
Inception Network
"""
import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


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


def inception_network():
    """
    builds the inception network as described in Going Deeper with
    Convolutions (2014)
    :return: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    conv2d = K.layers.Conv2D(64, 7, padding='same',
                             activation='relu', strides=2,
                             kernel_initializer='he_normal')(X)
    max_pooling2d = K.layers.MaxPool2D(3, 2, padding='same')(conv2d)
    conv2d_1 = K.layers.Conv2D(64, 1, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(max_pooling2d)
    conv2d_2 = K.layers.Conv2D(192, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal')(conv2d_1)
    max_pooling2d_1 = K.layers.MaxPool2D(3, 2, padding='same')(conv2d_2)
    block = inception_block(max_pooling2d_1, [64, 96, 128, 16, 32, 32])
    block_1 = inception_block(block, [128, 128, 192, 32, 96, 64])
    max_pooling2d_4 = K.layers.MaxPool2D(3, 2, padding='same')(block_1)
    block_2 = inception_block(max_pooling2d_4, [192, 96, 208, 16, 48, 64])
    block_3 = inception_block(block_2, [160, 112, 224, 24, 64, 64])
    block_4 = inception_block(block_3, [128, 128, 256, 24, 64, 64])
    block_5 = inception_block(block_4, [112, 144, 288, 32, 64, 64])
    block_6 = inception_block(block_5, [256, 160, 320, 32, 128, 128])
    max_pooling2d_10 = K.layers.MaxPool2D(3, 2, padding='same')(block_6)
    block_7 = inception_block(max_pooling2d_10, [256, 160, 320, 32, 128, 128])
    block_8 = inception_block(block_7, [384, 192, 384, 48, 128, 128])
    average_pooling2d = K.layers.AveragePooling2D(7, padding='same')(block_8)
    dropout = K.layers.Dropout(0.6)(average_pooling2d)
    dense = K.layers.Dense(1000)(dropout)
    model = K.models.Model(inputs=X, outputs=dense)
    return model
