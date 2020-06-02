#!/usr/bin/env python3
"""Optimize"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics
    :param network: model to optimize
    :param alpha: learning rate
    :param beta1: first Adam optimization parameter
    :param beta2: second Adam optimization parameter
    :return: None
    """
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
