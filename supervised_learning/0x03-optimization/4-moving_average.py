#!/usr/bin/env python3
"""
Moving Average
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set
    """
    ma = []
    v = 0
    t = 1
    for d in data:
        v = beta * v + (1 - beta) * d
        ma.append(v / (1 - beta**t))
        t += 1
    return ma
