#!/usr/bin/env python3
""" Bracing The Elements"""


def np_elementwise(mat1, mat2):
    """performs element-wise
    add, sub, mul, and div"""

    add = mat1.__add__(mat2)
    sub = mat1.__sub__(mat2)
    mul = mat1.__mul__(mat2)
    div = mat1.__truediv__(mat2)

    return add, sub, mul, div
