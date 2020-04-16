#!/usr/bin/env python3
"""The Whole Barn"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices
    along a specific axis"""
    mat1_shape = matrix_shape(mat1)
    mat2_shape = matrix_shape(mat2)

    if len(mat1_shape) != len(mat2_shape):
        return None
    arr1 = flat_mtx(mat1)
    arr2 = flat_mtx(mat2)

    final_shape = mat1_shape[:]
    final_shape[axis] = mat1_shape[axis] + mat2_shape[axis]

    print('s1: {}, s2: {}, fs: {}'.format(mat1_shape, mat2_shape, final_shape))
    arr = sort_arr(arr1, arr2, mat1_shape,
                   mat2_shape, final_shape, axis)
    return reshape(arr, final_shape)


def sort_arr(arr1, arr2, sh1, sh2, fs, ax):
    """Sort array following the new shape"""
    arr = []
    st1 = 0
    st2 = 0

    while st1 < len(arr1) and st2 < len(arr2):
        arr += arr1[st1:(st1+sh1[ax])]
        arr += arr2[st2:(st2+sh2[ax])]
        st1 += sh1[ax]
        st2 += sh2[ax]
    return arr


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
