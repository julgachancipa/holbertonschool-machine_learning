#!/usr/bin/env python3
"""Validate"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Train and analyze validaiton data
    :param network: model to train
    :param data: numpy.ndarray with input data
    :param labels: one-hot numpy.ndarray containing the labels of data
    :param batch_size: size of the batch used for mini-batch gradient descent
    :param epochs: number of passes through data for mini-batch
    gradient descent
    :param verbose: boolean that determines if output should be
    printed during training
    :param shuffle: boolean that determines whether to shuffle the
    batches every epoch
    :param validation_data: data to valid
    :return: History object generated after training the model

    if type(validation_data) is not tuple:
        validation_data = None
    """
    network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                verbose=verbose, shuffle=shuffle,
                validation_data=validation_data)

    return network.history
