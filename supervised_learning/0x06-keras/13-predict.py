#!/usr/bin/env python3
"""
Predict
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network
    :param network:
    :param data:
    :param verbose:
    :return:
    """
    return network.predict(data, verbose=verbose)
