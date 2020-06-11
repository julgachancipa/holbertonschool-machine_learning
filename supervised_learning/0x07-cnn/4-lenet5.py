#!/usr/bin/env python3
"""
LeNet-5
"""
import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using
    tensorflow
    :param x: tf.placeholder of shape (m, 28, 28, 1) containing the input
    images for the network
        m is the number of images
    :param y: tf.placeholder of shape (m, 10) containing the one-hot labels
    for the network
    :return:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default
        hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    k_init = tf.contrib.layers.variance_scaling_initializer()

    conv_1 = tf.layers.Conv2D(6, 5, padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=k_init)(x)
    pool_1 = tf.layers.MaxPooling2D(2, 2)(conv_1)
    conv_2 = tf.layers.Conv2D(16, 5, padding='valid',
                              activation=tf.nn.relu,
                              kernel_initializer=k_init)(pool_1)
    pool_2 = tf.layers.MaxPooling2D(2, 2)(conv_2)
    flat = tf.layers.Flatten()(pool_2)
    fully_1 = tf.layers.Dense(120,
                              activation=tf.nn.relu,
                              kernel_initializer=k_init)(flat)
    fully_2 = tf.layers.Dense(84,
                              kernel_initializer=k_init,
                              activation=tf.nn.relu)(fully_1)
    y_pred = tf.layers.Dense(10, kernel_initializer=k_init)(fully_2)
    # Calculate loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    # Calculate Accuracy
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    y_out = tf.nn.softmax(y_pred)
    return y_out, train, loss, accuracy
