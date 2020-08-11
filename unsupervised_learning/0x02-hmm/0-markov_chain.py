#!/usr/bin/env python3
"""
Markov Chain
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular state
    after a specified number of iterations
    :param P: is a square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    :param s: is a numpy.ndarray of shape (1, n) representing the probability
    of starting in each state
    :param t: is the number of iterations that the markov chain has been
    through
    :return: a numpy.ndarray of shape (1, n) representing the probability of
    being in a specific state after t iterations, or None on failure
    """
    s = s.astype(float)
    for _ in range(t):
        s = s @ P
    return s
