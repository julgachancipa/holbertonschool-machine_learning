#!/usr/bin/env python3
"""Train"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent
    :param network: model to train
    :param data: numpy.ndarray with input data
    :param labels: one-hot numpy.ndarray containing the labels of data
    :param batch_size: size of the batch used for mini-batch gradient descent
    :param epochs: number of passes through data for
    mini-batch gradient descent
    :param verbose: boolean that determines if output
    should be printed during training
    :param shuffle: boolean that determines whether to shuffle
    the batches every epoch
    :return: History object generated after training the model
    """
    network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                verbose=verbose, shuffle=shuffle)
    return network.history
