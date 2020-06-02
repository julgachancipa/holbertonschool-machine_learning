#!/usr/bin/env python3
"""Save and Load Model"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model
    :param network: model to save
    :param filename:  path of the file
    that the model should be saved to
    :return: None
    """
    network.save(filename)


def load_model(filename):
    """
    loads an entire model
    :param filename:  path of the file
    that the model should be loaded
    :return: None
    """
    return K.models.load_model(filename)
