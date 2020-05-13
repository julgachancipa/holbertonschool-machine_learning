#!/usr/bin/env python3
"""All the Activations"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path


def tanh(z):
    """Calculates the tanh function"""
    return np.sinh(z) / np.cosh(z)


def sigmoid(z):
    """Calculates the sigmoid function"""
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """takes as input a vector of K real numbers, and
    normalizes it into a probability distribution"""
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=0, keepdims=True)
    return a


class DeepNeuralNetwork:
    """defines a deep neural network with one hidden
    layer performing binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        """class constructor"""
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')
        if activation != 'sig' or activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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
    def activation(self):
        """acivation function"""
        return self.__activation

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
        for lay in range(self.__L):
            Al = self.__cache["A" + str(lay)]
            Wl = self.__weights["W" + str(lay + 1)]
            bl = self.__weights["b" + str(lay + 1)]
            Zl = np.matmul(Wl, Al) + bl
            if lay != self.__L - 1:
                if self.__activation == 'sig':
                    self.__cache["A" + str(lay + 1)] = sigmoid(Zl)
                elif self.__activation == 'tanh':
                    self.__cache["A" + str(lay + 1)] = tanh(Zl)
            else:
                self.__cache["A" + str(lay + 1)] = softmax(Zl)
        return self.__cache["A" + str(lay + 1)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the
        model using cross entropy"""
        m = (Y.shape[1])
        L = -np.sum(Y * np.log(A), axis=1, keepdims=True)
        return np.sum(L) / m

    def evaluate(self, X, Y):
        """Evaluates the neurons predictions"""
        AF, cache = self.forward_prop(X)
        A = np.where(AF == np.amax(AF, axis=0), 1, 0)
        return A, self.cost(Y, AF)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient
        descent on the neural network"""
        m = (Y.shape[1])
        Al = cache["A" + str(self.__L)]
        dAl = Al - Y
        for l in reversed(range(1, self.__L + 1)):
            Al = cache["A" + str(l)]
            if self.__activation == 'sig':
                gl_d = Al * (1 - Al)
            else:
                gl_d = 1 - np.power(Al, 2)
            dZl = dAl * gl_d
            Al_1 = cache["A" + str(l - 1)]
            dWl = (1/m) * np.matmul(dZl, Al_1.T)
            dbl = (1/m) * np.sum(dZl, axis=1, keepdims=True)
            Wl = self.__weights["W" + str(l)]
            dAl = np.matmul(Wl.T, dZl)

            kW = "W" + str(l)
            kb = "b" + str(l)
            self.__weights[kW] = self.__weights[kW] - alpha * dWl
            self.__weights[kb] = self.__weights[kb] - alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) != int:
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        g_x = []
        g_y = []
        for i in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            c = self.cost(Y, A)
            if not i % step:
                if verbose:
                    print('Cost after {} iterations: {}'.format(i, c))
                g_x.append(i)
                g_y.append(c)

        A, cost = self.evaluate(X, Y)
        if verbose:
            print('Cost after {} iterations: {}'.format(iterations, cost))
        g_x.append(iterations)
        g_y.append(cost)
        if graph:
            plt.plot(g_x, g_y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return A, cost

    def save(self, filename):
        """Saves the instance object
        to a file in pickle format"""
        if filename[-4:] != '.pkl':
            filename += '.pkl'
        outfile = open(filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if not os.path.exists(filename):
            return None
        infile = open(filename, 'rb')
        NN = pickle.load(infile)
        infile.close()
        return NN
