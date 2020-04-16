#!/usr/bin/env python3
"""The Whole Barn"""


def add_matrices(mat1, mat2):
    """Adds two matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    arr1 = flat_mtx(mat1)
    arr2 = flat_mtx(mat2)

    f_add = [arr1[i] + arr2[i] for i in range(len(arr1))]

    return reshape(f_add, matrix_shape(mat1))


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])


def flat_mtx(matrix):
    """Flat a matrix"""
    if not type(matrix) == list:
        return []

    if len(matrix_shape(matrix)) <= 1:
        return matrix

    mtx = matrix[:]
    for d in range(len(matrix_shape(matrix)) - 1):
        aux = []
        for row in mtx:
            for j in row:
                aux.append(j)
        mtx = aux

    return mtx


def reshape(lst, shape):
    """Reshape a list"""
    if len(shape) == 1:
        return lst
    n = reduce(mul, shape[1:])
    return [reshape(lst[i*n:(i+1)*n], shape[1:])
            for i in range(len(lst)//n)]


def reduce(function, iterable, initializer=None):
    """Reduce funcion"""
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value


def mul(a, b):
    """Mul a*b"""
    return a * b
