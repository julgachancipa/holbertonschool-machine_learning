#!/usr/bin/env python3
"""
Convolution with Channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels
    :param images: contain multiple images
    :param kernel: contain the kernel for the convolution
    :param padding: padding
    :param stride: tuple of (sh, sw)
    :return: numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
        oh = h
        ow = w

    elif padding == 'valid':
        oh = int(np.ceil((h - kh + 1) / sh))
        ow = int(np.ceil((w - kw + 1) / sw))

    else:
        ph = padding[0]
        pw = padding[1]
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
        oh = ((h - kh + 2 * ph) // sh) + 1
        ow = ((w - kw + 2 * pw) // sw) + 1

    conv = np.zeros((m, oh, ow))
    row = 0
    for i in range(oh):
        col = 0
        for j in range(ow):
            aux = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel
            conv[:, row, col] = np.sum(aux, axis=(1, 2, 3))
            col += 1
        row += 1
    return conv
