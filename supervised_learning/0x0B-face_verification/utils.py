#!/usr/bin/env python3
"""
Load Images
"""
import cv2
import glob
import csv
import os
import numpy as np


def load_images(images_path, as_array=True):
    """
    Loads images from a directory or file

    :param images_path: is the path to a directory from which to load images
    :param as_array: is a boolean indicating whether the images should
    be loaded as one numpy.ndarray

    :return: images, filenames
    """
    images, filenames = [], []

    files = glob.glob(images_path + '/*.jpg')
    files.sort()

    for i, file in enumerate(files):
        img_BGR = cv2.imread(file)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        if as_array:
            if i is 0:
                images = [img_RGB]
            else:
                images = np.concatenate((images, [img_RGB]))
        else:
            images.append(img_RGB)

        filenames.append(file.split('/')[-1])

    return images, filenames


def load_csv(csv_path, params={}):
    """
    loads the contents of a csv file as a list of lists

    :param csv_path: path to the csv to load
    :param params: parameters to load the csv with

    :return: a list of lists representing the contents found in csv_path
    """
    csv_list = []
    with open(csv_path, newline='') as csvfile:
        content = csv.reader(csvfile, params)
        for row in content:
            csv_list.append(row)
    return csv_list


def save_images(path, images, filenames):
    """
    saves images to a specific path

    :param path: path to the directory in which the images should be saved
    :param images: list/numpy.ndarray of images to save
    :param filenames: list of filenames of the images to save

    :return: True on success and False on failure
    """
    for img, filename in zip(images, filenames):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        success = cv2.imwrite(os.path.join(path, filename), img_RGB)
    return success


def generate_triplets(images, filenames, triplet_names):
    """
    Generates triplets

    :param images: numpy.ndarray of shape (i, n, n, 3) containing the aligned
    images in the dataset
    :param filenames: list of length i containing the corresponding filenames
    for images
    :param triplet_names: list of length m of lists where each sublist
    contains the filenames of an anchor, positive, and negative image

    :return: [A, P, N]
        A is a numpy.ndarray of shape (m, n, n, 3) containing the anchor
        images for all m triplets
        P is a numpy.ndarray of shape (m, n, n, 3) containing the positive
        images for all m triplets
        N is a numpy.ndarray of shape (m, n, n, 3) containing the negative
        images for all m triplets
    """
    img_dict = dict(zip(filenames, images))
    for i, triplet in enumerate(triplet_names):
        if i is 0:
            A = [img_dict[triplet[0] + '.jpg']]
            P = [img_dict[triplet[1] + '.jpg']]
            N = [img_dict[triplet[2] + '.jpg']]
        else:
            A = np.concatenate((A, [img_dict[triplet[0] + '.jpg']]))
            P = np.concatenate((P, [img_dict[triplet[1] + '.jpg']]))
            N = np.concatenate((N, [img_dict[triplet[2] + '.jpg']]))
    return A, P, N
