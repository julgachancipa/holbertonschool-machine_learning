#!/usr/bin/env python3
"""
Same Convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    :param images: multiple grayscale images
    :param kernel: kernel for the convolution
    :return: numpy.ndarray containing the convolved images
    """
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    images = np.pad(images, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    hc = ih - kh + 1
    wc = iw - kw + 1

    conv = np.zeros((m, hc, wc))
    for h in range(hc):
        for w in range(wc):
            aux = np.multiply(images[:, h:h+kh, w:w+kw], kernel)
            conv[:, h, w] = np.sum(aux, axis=(1, 2))
    return conv
