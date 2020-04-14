#!/usr/bin/env python3
"""Line Up"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise"""
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
    else:
        return None
