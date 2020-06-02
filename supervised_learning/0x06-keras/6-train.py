#!/usr/bin/env python3
"""Early Stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
     train the model using early stopping
    :param network: model to train
    :param data: numpy.ndarray with input data
    :param labels: one-hot numpy.ndarray containing
    the labels of data
    :param batch_size: size of the batch used for
    mini-batch gradient descent
    :param epochs: number of passes through data for
    mini-batch gradient descent
    :param verbose: boolean that determines if output
    should be printed during training
    :param shuffle: boolean that determines whether to
    shuffle the batches every epoch
    :param validation_data: data to valid
    :param early_stopping: boolean that indicates whether early
    stopping should be used
    :param patience: patience used for early stopping
    :return: History object generated after training the model
    """
    if early_stopping and validation_data:
        callbacks = K.callbacks.EarlyStopping(patience=patience)
    else:
        callbacks = None
    network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                verbose=verbose, shuffle=shuffle,
                validation_data=validation_data,
                callbacks=[callbacks])
    return network.history
