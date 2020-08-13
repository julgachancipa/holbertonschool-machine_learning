#!/usr/bin/env python3
"""
The Viretbi Algorithm
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    markov model

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
        Transition[i, j] is the probability of transitioning from
        the hidden state i to j

    :param Initial: a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state

    :return: path, P, or None, None on failure
        path is the a list of length T containing the most likely sequence of
        hidden states
        P is the probability of obtaining the path sequence
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    bp = np.zeros((N, T))

    v = np.zeros((N, T))
    v[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for s in range(N):
            aux = (v[:, t - 1] * Transition[:, s]) * \
                   Emission[s, Observation[t]]
            v[s, t] = np.max(aux)

            bp[s, t] = np.argmax((v[:, t - 1] * Transition[:, s]) *
                                 Emission[s, Observation[t]])
    P = np.max(v[:, -1])

    S = np.argmax(v[:, -1])

    path = [S]

    for t in range(T - 1, 0, -1):
        S = int(bp[S, t])
        path.append(S)

    return path[::-1], P
