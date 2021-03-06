#!/usr/bin/env python3
"""
Same Convolution
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding
    :param images: multiple grayscale images
    :param kernel: kernel for the convolution
    :param padding:  tuple of (ph, pw)
    :return: numpy.ndarray containing the convolved images
    """
    images = np.pad(images, pad_width=((0, 0), (padding[0], padding[0]),
                    (padding[1], padding[1])), mode='constant',
                    constant_values=0)
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
            aux = np.multiply(images[:, h:h + kh, w:w + kw], kernel)
            aux = np.reshape(aux, (m, kh * kw))
            conv[:, h, w] = np.sum(aux, axis=1)
    return conv
