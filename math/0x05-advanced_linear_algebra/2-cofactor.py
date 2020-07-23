#!/usr/bin/env python3
"""
Cofactor
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    :param matrix: list of lists whose determinant should be calculated
    :return: the determinant of matrix
    """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError('matrix must be a list of lists')
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if len(matrix) is 1 and len(matrix[0]) is 0:
        return 1
    if len(matrix) is not len(matrix[0]):
        raise ValueError('matrix must be a square matrix')
    if len(matrix) is 1 and len(matrix[0]) is 1:
        return matrix[0][0]

    if len(matrix) is 2 and len(matrix[0]) is 2:
        det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return det

    total = 0
    for j in range(len(matrix)):
        tmp = [x[:] for x in matrix]
        tmp = tmp[1:]
        height = len(tmp)

        for i in range(height):
            tmp[i] = tmp[i][0:j] + tmp[i][j+1:]
        sign = (-1) ** (j % 2)
        sub_det = determinant(tmp)
        total += sign * matrix[0][j] * sub_det
    return total


def getMatrixMinor(m, i, j):
    """
    :param m: matrix
    :param i: x coord.
    :param j: y coord.
    :return: minor of the position
    """
    return determinant([row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])])


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    :param matrix: list of lists whose minor matrix should be calculated
    :return: minor matrix of matrix
    """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError('matrix must be a list of lists')
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix is [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    if not all(len(matrix) == col for col in [len(row) for row in matrix]):
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix) is 1 and len(matrix[0]) is 1:
        return [[1]]
    minors = [x[:] for x in matrix]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            minors[i][j] = getMatrixMinor(matrix, i, j)
    return minors


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    :param matrix: list of lists whose cofactor matrix should be calculated
    :return: cofactor matrix of matrix
    """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError('matrix must be a list of lists')
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix is [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    if not all(len(matrix) == col for col in [len(row) for row in matrix]):
        raise ValueError('matrix must be a non-empty square matrix')
    cofactors = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            cofactors[i][j] *= (-1) ** (i+j)
    return cofactors
