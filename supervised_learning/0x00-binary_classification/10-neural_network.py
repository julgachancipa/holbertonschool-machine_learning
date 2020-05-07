#!/usr/bin/env python3
"""Privatize NeuralNetwork"""
import numpy as np


def sigmoid(z):
    """Calculates the sigmoid function"""
    return 1 / (1 + np.exp(-z))


class NeuralNetwork:
    """defines a neural network with one hidden
    layer performing binary classification"""

    def __init__(self, nx, nodes):
        """class constructor"""
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) != int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """weights vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """bias for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """activated output for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(Z1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = sigmoid(Z2)
        return self.__A1, self.__A2
