#!/usr/bin/env python3
"""
Transfer Knowledge
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    pre-processes the data for your model
    """
    X_scaled = X.astype('float32')
    Y_scaled = Y.astype('float32')
    X_scaled /= 255
    Y_scaled /= 255
    print(X.shape, Y.shape)
