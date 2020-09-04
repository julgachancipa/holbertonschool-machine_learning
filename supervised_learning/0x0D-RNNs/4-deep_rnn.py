#!/usr/bin/env python3
"""
Deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    :param rnn_cells: is a list of RNNCell instances of length l
    that will be used for the forward propagation
        l is the number of layers
    :param X: is the data to be used, given as a numpy.ndarray of
    shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    :param h_0: is the initial hidden state, given as a numpy.ndarray
    of shape (l, m, h)
        h is the dimensionality of the hidden state
    :return: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    T, m, i = X.shape
    L, _, h = h_0.shape

    H = np.zeros((T + 1, L, m, h))
    H[0] = h_0
    Y = []
    for t in range(T):
        aux = X[t]
        for lay in range(L):
            h_n, y = rnn_cells[lay].forward(H[t, lay], aux)
            H[t + 1, lay] = h_n
            aux = h_n
        Y.append(y)
    return H, np.array(Y)
