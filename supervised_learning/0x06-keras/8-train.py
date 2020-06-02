#!/usr/bin/env python3
"""Save the best"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
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
    :param learning_rate_decay: boolean that indicates whether
    learning rate decay should be used
    :param alpha: the initial learning rate
    :param decay_rate: the decay rate
    :param save_best: boolean indicating whether to save the model
    after each epoch if it is the best
    :param filepath: file path where the model should be saved
    :return: History object generated after training the model
    """
    def schedule(epoch):
        """function that takes an epoch index as input
        and returns a new learning rate as output"""
        return alpha / (1 + (decay_rate * epoch))

    callbacks = []
    if early_stopping and validation_data:
        callbacks += [K.callbacks.EarlyStopping(patience=patience)]
    if learning_rate_decay and validation_data:
        callbacks += [K.callbacks.LearningRateScheduler(schedule, verbose=1)]
    if save_best and validation_data:
        callbacks += [K.callbacks.ModelCheckpoint(filepath, mode='min',
                                                  save_best_only=True)]
    if len(callbacks) < 1:
        callbacks = None
    network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                verbose=verbose, shuffle=shuffle,
                validation_data=validation_data,
                callbacks=callbacks)
    return network.history
