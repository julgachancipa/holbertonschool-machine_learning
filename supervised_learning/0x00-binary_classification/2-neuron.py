#!/usr/bin/env python3
"""Neuron Forward Propagation"""
import numpy as np


def sigmoid(z):
    """Calculate the sigmoid function"""
    return 1/(1+np.exp(-z))


class Neuron:
    """defines a single neuron performing
    binary classification"""

    def __init__(self, nx):
        """class constructor"""
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """weights vector for the neuron"""
        return self.__W

    @property
    def b(self):
        """bias for the neuron"""
        return self.__b

    @property
    def A(self):
        """activated output of the neuron (prediction)"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = sigmoid(z)
        return self.__A
