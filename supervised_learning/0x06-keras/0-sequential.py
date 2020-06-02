#!/usr/bin/env python3
"""Sequential"""
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
    model = K.Sequential()
    L = len(layers)
    for l in range(L):
        if l is 0:
            model.add(K.layers.Dense(layers[l], input_shape=(nx,),
                                     kernel_regularizer=K.regularizers.
                                     l2(lambtha),
                                     activation=activations[l]))
        else:
            model.add(K.layers.Dense(layers[l],
                                     kernel_regularizer=K.regularizers.
                                     l2(lambtha),
                                     activation=activations[l]))
        if l is not L - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
