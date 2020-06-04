#!/usr/bin/env python3
"""
Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images
    :param images: contain multiple grayscale images
    :param kernel: contain the kernel for the convolution
    :param padding: either a tuple of (ph, pw), ‘same’, or ‘valid’
    :param stride: tuple of (sh, sw)
    :return:
    """
    if padding == 'same':
        images = np.pad(images, pad_width=((0, 0), (1, 1), (1, 1)),
                        mode='constant', constant_values=0)
    elif padding != 'valid':
        ph = padding[0]
        pw = padding[1]
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    hc = (ih - kh + 1)
    wc = (iw - kw + 1)

    conv = np.zeros((m, hc // stride[0], wc // stride[1]))

    i = 0
    for h in range(0, hc, stride[0]):
        j = 0
        for w in range(0, wc, stride[1]):
            aux = np.multiply(images[:, h:h + kh, w:w + kw], kernel)
            aux = np.reshape(aux, (m, kh * kw))
            conv[:, i, j] = np.sum(aux, axis=1)
            j += 1
        i += 1
    return conv
