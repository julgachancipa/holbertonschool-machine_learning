#!/usr/bin/env python3
"""
LeNet-5
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras
    :param X:
    :return:
    """
    conv_1 = K.layers.Conv2D(6, 5, padding='same')(X)
    pool_1 = K.layers.MaxPool2D(2, 2)(conv_1)
    conv_2 = K.layers.Conv2D(16, 5, padding='valid')(pool_1)
    pool_2 = K.layers.MaxPool2D(2, 2)(conv_2)
    flat = K.layers.Flatten()(pool_2)
    f_con_1 = K.layers.Dense(120, input_shape=X.shape,
                             activation='relu',
                             kernel_initializer='he_normal')(flat)
    f_con_2 = K.layers.Dense(84, activation='relu',
                             kernel_initializer='he_normal')(f_con_1)
    Y = K.layers.Dense(10, activation='softmax',
                       kernel_initializer='he_normal')(f_con_2)
    model = K.Model(X, Y)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
