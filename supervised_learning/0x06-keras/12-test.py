#!/usr/bin/env python3
"""Test"""


def test_model(network, data, labels, verbose=True):
    """
    tests a neural network
    :param network: network model to test
    :param data: input data to test the model with
    :param labels: correct one-hot labels of data
    :param verbose:  boolean that determines if output
    should be printed during the testing process
    :return: loss and accuracy of the model with the
    testing data, respectively
    """
    return network.evaluate(data, labels, verbose=verbose)
