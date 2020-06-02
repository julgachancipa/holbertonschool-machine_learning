#!/usr/bin/env python3
"""Input"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    :param nx: number of input features to the network
    :param layers: list containing the number of nodes in
    each layer of the network
    :param activations:  list containing the activation
    functions used for each layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout
    :return: the keras model
    """
    x = K.Input(shape=(nx,))
    L = len(layers)
    for l in range(L):
        if l is 0:
            y = K.layers.Dense(layers[l],
                               kernel_regularizer=K.regularizers.l2(lambtha),
                               activation=activations[l])(x)
        else:
            y = K.layers.Dense(layers[l],
                               kernel_regularizer=K.regularizers.l2(lambtha),
                               activation=activations[l])(y)
        if l is not L - 1:
            y = K.layers.Dropout(1 - keep_prob)(y)
    return K.Model(x, y)
