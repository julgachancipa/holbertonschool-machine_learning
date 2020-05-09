#!/usr/bin/env python3
"""DeepNeuralNetwork Cost"""
import numpy as np


def sigmoid(z):
    """Calculates the sigmoid function"""
    return 1 / (1 + np.exp(-z))


class DeepNeuralNetwork:
    """defines a deep neural network with one hidden
    layer performing binary classification"""

    def __init__(self, nx, layers):
        """class constructor"""
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if type(layers[i]) != int or layers[i] < 0:
                raise TypeError('layers must be a list of positive integers')
            kb = "b" + str(i + 1)
            kW = "W" + str(i + 1)

            self.__weights[kb] = np.zeros(layers[i]).reshape(layers[i], 1)
            if i > 0:
                aux = layers[i-1]
            else:
                aux = nx
            self.__weights[kW] = np.random.randn(layers[i],
                                                 aux) * np.sqrt(2/aux)

    @property
    def L(self):
        """number of layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """weights and biased of the network"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation
        of the neural network"""
        self.__cache["A0"] = X
        for l in range(self.__L):
            Al = self.__cache["A" + str(l)]
            Wl = self.__weights["W" + str(l + 1)]
            bl = self.__weights["b" + str(l + 1)]
            Zl = np.matmul(Wl, Al) + bl
            self.__cache["A" + str(l + 1)] = sigmoid(Zl)
        return self.__cache["A" + str(l + 1)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model
        using logistic regression"""
        m = (Y.shape[1])
        return -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
