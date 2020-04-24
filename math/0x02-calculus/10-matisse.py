#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if type(poly) != list:
        return None
    if poly == []:
        return None
    if len(poly) == 1:
        return [0]
    result = []
    for i in range(1, len(poly)):
        result.append(i * poly[i])

    if result == [0] * len(result):
        return [0]

    return result
