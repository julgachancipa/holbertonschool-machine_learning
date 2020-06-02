#!/usr/bin/env python3
"""Save and Load Configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format
    :param network: model whose configuration should be saved
    :param filename: path of the file that the configuration
    should be saved to
    :return: None
    """
    json_string = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(json_string)


def load_config(filename):
    """
    loads a model with a specific configuration
    :param filename: path of the file containing the model’s
    configuration in JSON format
    :return: loaded model
    """
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)
    return loaded_model
