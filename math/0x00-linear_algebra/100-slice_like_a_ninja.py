#!/usr/bin/env python3
"""Slice Like A Ninja"""


def np_slice(matrix, axes={}):
    """Slices a matrix along
    a specific axes"""
    new_mtx = matrix.copy()

    ix = []
    for dim in range(new_mtx.ndim):
        ax = axes.get(dim, None)

        if ax is None:
            ix.append(slice(ax))
        else:
            ix.append(slice(*ax))

    return new_mtx[tuple(ix)]
