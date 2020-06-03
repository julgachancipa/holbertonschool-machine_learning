#!/usr/bin/env python3
"""
Valid Convolution
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images
    :param images: multiple grayscale images
    :param kernel: kernel for the convolution
    :return: numpy.ndarray containing the convolved images
    """
    hc = images.shape[1] - kernel.shape[0] + 1
    wc = images.shape[2] - kernel.shape[1] + 1
    
    for w in range(wc):
        for h in range(hc):