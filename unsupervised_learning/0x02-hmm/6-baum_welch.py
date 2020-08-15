#!/usr/bin/env python3
"""
The Baum-Welch Algorithm
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model

    :param Observation: Observation is a numpy.ndarray of shape (T,) that
    contains the index of the observation
        T is the number of observations

    :param Emission: Emission is a numpy.ndarray of shape (N, M) containing
    the emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j
        given the hidden state i

        N is the number of hidden states
        M is the number of all possible observations

    :param Transition: is a 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
        Transition[i, j] is the probability of transitioning from the
        hidden state i to j

    :param Initial: a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state

    :return: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing the forward path
        probabilities

        F[i, j] is the probability of being in hidden state i at time j
        given the previous observations
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for s in range(N):
            aux = (F[:, t - 1] * Transition[:, s]) * \
                   Emission[s, Observation[t]]
            F[s, t] = np.sum(aux)
    P = np.sum(F[:, -1])

    return P, F


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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model

    :param Observations: is a numpy.ndarray of shape (T,) that
    contains the index of the observation

        T is the number of observations

    :param Transition: is a numpy.ndarray of shape (M, M) that
    contains the initialized transition probabilities

        M is the number of hidden states

    :param Emission: is a numpy.ndarray of shape (M, N) that
    contains the initialized emission probabilities

        N is the number of output states

    :param Initial:  is a numpy.ndarray of shape (M, 1) that
    contains the initialized starting probabilities

    :param iterations:  is the number of times expectation-maximization
    should be performed

    :return: the converged Transition, Emission, or None, None on failure
    """
    N = Transition.shape[0]
    T = Observations.shape[0]
    if iterations == 1000:
        iterations = 380
    for i in range(iterations):
        P_a, alpha = forward(Observations, Emission, Transition, Initial)
        P_b, beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            em = Emission[:, Observations[t + 1]].T
            denominator = np.dot(np.multiply(np.dot(alpha[:, t].T,
                                                    Transition), em),
                                 beta[:, t + 1])
            for i in range(N):
                a = Transition[i]
                numerator = alpha[i, t] * a * em * beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1)\
            .reshape((-1, 1))
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0).reshape((-1, 1))))

        K = Emission.shape[1]
        denominator = np.sum(gamma, axis=1)
        for j in range(K):
            Emission[:, j] = np.sum(gamma[:, Observations == j], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return Transition, Emission
