#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    :param input_dims: is a tuple of integers containing the dimensions
    of the model input
    :param filters: is a list containing the number of filters for each
    convolutional layer in the encoder, respectively
    :param latent_dims: is a tuple of integers containing the dimensions
    of the latent space representation
    :return:
    """
    input_encoder = keras.layers.Input(shape=input_dims)
    input_encoded = input_encoder

    for i, n in enumerate(filters):
        encoded = keras.layers.Conv2D(n, (3, 3), activation='relu',
                                      padding='same')(input_encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
        input_encoded = encoded

    encoder = keras.models.Model(input_encoder, encoded)

    input_decoder = keras.layers.Input(shape=latent_dims)
    input_decoded = input_decoder

    for i, n in enumerate(filters[::-1]):
        if i == len(filters) - 1:
            decoded = keras.layers.Conv2D(n, (3, 3), activation='relu',
                                          padding='valid')(input_decoded)
        else:
            decoded = keras.layers.Conv2D(n, (3, 3), activation='relu',
                                          padding='same')(input_decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
        input_decoded = decoded

    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                  activation='sigmoid',
                                  padding='same')(decoded)

    decoder = keras.models.Model(input_decoder, decoded)

    input_auto = keras.layers.Input(shape=input_dims)
    encoderOut = encoder(input_auto)
    decoderOut = decoder(encoderOut)
    auto = keras.models.Model(input_auto, decoderOut)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
