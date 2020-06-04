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
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        oh = np.ceil(h/sh)
        ow = np.ceil(w/sw)
        ph = max((oh - 1) * sh + kh, 0)
        pw = max((ow - 1) * sw + kw, 0)
        ptb = np.floor(ph/2)
        plr = np.floor(pw/2)
        images = np.pad(images, pad_width=((0, 0), (ptb, ptb), (plr, plr)),
                        mode='constant', constant_values=0)
    elif padding != 'valid':
        ph = padding[0]
        pw = padding[1]
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
    hc = (h - kh + 1)
    wc = (w - kw + 1)

    conv = np.zeros((m, hc // stride[0], wc // stride[1]))

    i = 0
    for h_i in range(0, hc, stride[0]):
        j = 0
        for w_i in range(0, wc, stride[1]):
            aux = np.multiply(images[:, h_i:h_i + kh, w_i:w_i + kw], kernel)
            aux = np.reshape(aux, (m, kh * kw))
            conv[:, i, j] = np.sum(aux, axis=1)
            j += 1
        i += 1
    return conv
