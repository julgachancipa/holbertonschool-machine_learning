#!/usr/bin/env python3
"""Train NeuralNetwork"""
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

    def cost(self, Y, A):
        """Calculates the cost of the model
        using logistic regression"""
        m = (Y.shape[1])
        return -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A1, A2 = self.forward_prop(X)
        c = self.cost(Y, A2)
        A = np.round(A2)
        return A.astype(int), c

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = (X.shape[1])

        dZ2 = A2 - Y
        dW2 = (1/m) * np.matmul(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        g1_d = A1 * (1 - A1)

        aux1 = np.matmul(self.__W2.T, dZ2)
        aux2 = g1_d
        dZ1 = np.multiply(aux1, aux2)

        dW1 = (1/m) * np.matmul(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
