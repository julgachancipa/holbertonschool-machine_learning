#!/usr/bin/env python3
"""
The Backward Algorithm
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model

    :param Observation: is a numpy.ndarray of shape (T,) that contains
    the index of the observation
        T is the number of observations

    :param Emission: is a numpy.ndarray of shape (N, M) containing the
    emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given
        the hidden state i

        N is the number of hidden states
        M is the number of all possible observations

    :param Transition: is a 2D numpy.ndarray of shape (N, N) containing
    the transition probabilities
        Transition[i, j] is the probability of transitioning from the
        hidden state i to j

    :param Initial: a numpy.ndarray of shape (N, 1) containing the
    probability of starting in a particular hidden state

    :return: P, B, or None, None on failure
        Pis the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward
        path probabilities

        B[i, j] is the probability of generating the future observations
        from hidden state i at time j
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))

    for i in range(T - 2, -1, -1):
        for j in range(N):
            aux = B[:, i + 1] * Transition[j, :] *\
                  Emission[:, Observation[i + 1]]
            B[j, i] = np.sum(aux)

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
