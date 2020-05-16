#!/usr/bin/env python3
import tensorflow as tf
"""
Forward Propagation
"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation
    graph for the neural network
    """
    prev = x
    for i in range(len(layer_sizes)):
        n = layer_sizes[i]
        activation = activations[i]
        layer = create_layer(prev, n, activation)
        prev = layer
    return layer
