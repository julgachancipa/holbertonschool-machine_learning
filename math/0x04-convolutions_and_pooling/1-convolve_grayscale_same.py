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
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    hc = ih - kh + 1
    wc = iw - kw + 1

    pad_h = max((hc - 1) + kh - ih, 0)
    pad_w = max((wc - 1) + kw - iw, 0)
    p_t = pad_h // 2
    p_l = pad_w // 2
    images = np.pad(images, pad_width=((0, 0), (p_t, pad_h - p_t),
                                       (p_l, pad_w - p_l)), mode='constant')
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
