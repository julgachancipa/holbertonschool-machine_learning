#!/usr/bin/env python3
"""Size me please"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
