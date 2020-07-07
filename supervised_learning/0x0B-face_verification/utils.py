#!/usr/bin/env python3
"""
Load Images
"""
import cv2
import glob
import numpy as np


def load_images(images, as_array=True):
    """
    loads images from a directory or file
    :param images: is the path to a directory from which to load images
    :param as_array: is a boolean indicating whether the images should
    be loaded as one numpy.ndarray
    :return: images, filenames
    """
