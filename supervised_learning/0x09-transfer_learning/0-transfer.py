#!/usr/bin/env python3
"""
Transfer Knowledge
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    pre-processes the data for your model
    :param X: is a numpy.ndarray of shape (m, 32, 32, 3) containing the
    CIFAR 10 data, where m is the number of data points
    :param Y: Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
    labels for X
    :return: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X = X.astype('float32')
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return(X_p, Y_p)


(Xt, Yt), (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(Xt, Yt)
Xv_p, Yv_p = preprocess_data(X, Y)

# print(X_p.shape, Y_p.shape)
base_model = K.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                        pooling='avg', classes=Y_p.shape[1])

model = K.Sequential()
model.add(K.layers.UpSampling2D())
model.add(base_model)
model.add(K.layers.Flatten())
model.add(K.layers.Dense(512, activation=('relu')))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.Dense(256, activation=('relu')))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.Dense(10, activation=('softmax')))
callback = []


def decay(epoch):
    """ This method create the alpha"""
    return 0.001 / (1 + 1 * 30)


callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=X_p, y=Y_p,
                    batch_size=128,
                    validation_data=(Xv_p, Yv_p),
                    epochs=10, shuffle=True,
                    callbacks=callback,
                    verbose=1)
model.save('cifar10.h5')
