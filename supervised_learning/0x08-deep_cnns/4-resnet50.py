#!/usr/bin/env python3
"""
ResNet-50
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture as described in Deep Residual Learning
    for Image Recognition (2015)
    :return: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    conv2d = K.layers.Conv2D(64, 7, padding='same', strides=2,
                             kernel_initializer='he_normal')(X)

    batch_norm = K.layers.BatchNormalization()(conv2d)
    act = K.layers.Activation('relu')(batch_norm)
    max_pool = K.layers.MaxPool2D(3, 2, padding='same',)(act)

    proj_1 = projection_block(max_pool, [64, 64, 256], s=1)
    iden_1 = identity_block(proj_1, [64, 64, 256])
    iden_2 = identity_block(iden_1, [64, 64, 256])

    proj_2 = projection_block(iden_2, [128, 128, 512], s=2)
    iden_3 = identity_block(proj_2, [128, 128, 512])
    iden_4 = identity_block(iden_3, [128, 128, 512])
    iden_5 = identity_block(iden_4, [128, 128, 512])

    proj_3 = projection_block(iden_5, [256, 256, 1024], s=2)
    iden_6 = identity_block(proj_3, [256, 256, 1024])
    iden_7 = identity_block(iden_6, [256, 256, 1024])
    iden_8 = identity_block(iden_7, [256, 256, 1024])
    iden_9 = identity_block(iden_8, [256, 256, 1024])
    iden_10 = identity_block(iden_9, [256, 256, 1024])

    proj_4 = projection_block(iden_10, [512, 512, 2048], s=2)
    iden_11 = identity_block(proj_4, [512, 512, 2048])
    iden_12 = identity_block(iden_11, [512, 512, 2048])

    average_pool = K.layers.AveragePooling2D(7)(iden_12)
    dense = K.layers.Dense(1000, activation='softmax')(average_pool)
    model = K.models.Model(inputs=X, outputs=dense)
    return model
