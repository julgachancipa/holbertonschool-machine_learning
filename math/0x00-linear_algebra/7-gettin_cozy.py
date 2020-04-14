#!/usr/bin/env python3
"""Gettin’ Cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    m1 = [x[:] for x in mat1]
    m2 = [x[:] for x in mat2]
    if axis == 0:
        return m1 + m2
    else:
        return [m1[i] + m2[i] for i in range(len(m1))]
