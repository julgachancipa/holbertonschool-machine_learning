#!/usr/bin/env python3
"""
GRU Cell
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        :param i: is the dimensionality of the data
        :param h: is the dimensionality of the hidden state
        :param o: is the dimensionality of the outputs
        """
        self.Wz = np.random.randn(h+i, h)
        self.Wr = np.random.randn(h+i, h)
        self.Wh = np.random.randn(h+i, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, z):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """Compute softmax values for each sets of scores in x"""
        e_z = np.exp(z)
        return e_z / e_z.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        :param h_prev: is a numpy.ndarray of shape (m, h) containing the
        previous hidden state
        :param x_t: is a numpy.ndarray of shape (m, i) that contains the
        data input for the cell
            m is the batche size for the data
        :return: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        ug = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)
        rg = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        concat2 = np.concatenate((rg * h_prev, x_t), axis=1)
        cct = np.tanh(np.matmul(concat2, self.Wh) + self.bh)

        h_next = ug * cct + (1 - ug) * h_prev
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
