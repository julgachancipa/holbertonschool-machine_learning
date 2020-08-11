#!/usr/bin/env python3
"""
Agglomerative
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset
    :param X: is a numpy.ndarray of shape (n, d) containing the dataset
    :param dist: is the maximum cophenetic distance for all clusters
    :return: clss, a numpy.ndarray of shape (n,) containing the cluster
    indices for each data point
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                            criterion="distance")

    # Show figure
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    return clss
