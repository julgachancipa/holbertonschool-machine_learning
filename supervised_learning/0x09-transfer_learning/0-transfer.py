#!/usr/bin/env python3
"""
Transfer Knowledge
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    pre-processes the data for your model
    :param X: is a numpy.ndarray of shape (m, 32, 32, 3) containing the
    CIFAR 10 data, where m is the number of data points
    :param Y: Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
    labels for X
    :return: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = X.astype('float32')
    X_p /= 255

    Y_p= K.utils.to_categorical(Y)

    return X_p, Y_p


if __name__ == "__main__":
    base_inception = K.applications.inceptionV3(weights='imagenet', include_top=False)
