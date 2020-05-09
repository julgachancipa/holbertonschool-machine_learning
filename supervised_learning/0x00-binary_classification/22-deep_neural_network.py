#!/usr/bin/env python3
"""Train DeepNeuralNetwork"""
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

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        AF, cache = self.forward_prop(X)
        c = self.cost(Y, AF)
        A = np.round(AF)
        return A.astype(int), c

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient
        descent on the neural network"""
        m = (Y.shape[1])
        Al = cache["A" + str(self.__L)]
        dAl = -(Y/Al) + (1-Y)/(1-Al)
        for l in reversed(range(1, self.__L + 1)):
            Al = cache["A" + str(l)]
            gl_d = Al * (1 - Al)
            dZl = np.multiply(dAl, gl_d)
            Al_1 = cache["A" + str(l - 1)]
            dWl = (1/m) * np.matmul(dZl, Al_1.T)
            dbl = (1/m) * np.sum(dZl, axis=1, keepdims=True)
            Wl = self.__weights["W" + str(l)]
            dAl = np.matmul(Wl.T, dZl)

            kW = "W" + str(l)
            kb = "b" + str(l)
            self.__weights[kW] = self.__weights[kW] - alpha * dWl
            self.__weights[kb] = self.__weights[kb] - alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the deep neural network"""
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            _, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

        return self.evaluate(X, Y)
