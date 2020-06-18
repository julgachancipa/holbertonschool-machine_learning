#!/usr/bin/env python3
"""
DenseNet-121
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks
    :param growth_rate: is the growth rate
    :param compression:
    :return: is the compression factor
    """
    X = K.Input(shape=(224, 224, 3))
    batch_norm = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(batch_norm)
    conv2d = K.layers.Conv2D(64, 7, padding='same', strides=2,
                             kernel_initializer='he_normal')(act)
    max_pool = K.layers.MaxPool2D(3, 2, padding='same')(conv2d)
    d_block_1, nw_nb = dense_block(max_pool, 64, growth_rate, 6)
    t_lay_1, nw_nb = transition_layer(d_block_1, int(nw_nb), compression)
    d_block_2, nw_nb = dense_block(t_lay_1, int(nw_nb), growth_rate, 12)
    t_lay_2, nw_nb = transition_layer(d_block_2, int(nw_nb), compression)
    d_block_3, nw_nb = dense_block(t_lay_2, int(nw_nb), growth_rate, 24)
    t_lay_3, nw_nb = transition_layer(d_block_3, int(nw_nb), compression)
    d_block_4, nw_nb = dense_block(t_lay_3, int(nw_nb), growth_rate, 16)

    average_pool = K.layers.AveragePooling2D(7)(d_block_4)
    dense = K.layers.Dense(1000, activation='softmax')(average_pool)

    model = K.models.Model(inputs=X, outputs=dense)
    return model
