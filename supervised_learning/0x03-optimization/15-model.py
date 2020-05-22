#!/usr/bin/env python3
"""
Put it all together
"""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """
    returns two placeholders, x and y,
    for the neural network
    """
    x = tf.placeholder("float", [None, nx], name='x')
    y = tf.placeholder("float", [None, classes], name='y')

    return x, y


def create_layer(prev, n, activation):
    """
    Create Layer
    """
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=tf.contrib.layers.
                            variance_scaling_initializer(mode="FAN_AVG"))
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural
    """
    lay = tf.layers.Dense(units=n,
                          kernel_initializer=tf.contrib.layers.
                          variance_scaling_initializer(mode="FAN_AVG"))
    z = lay(prev)

    mean, variance = tf.nn.moments(z, [0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    z_n = tf.nn.batch_normalization(z, mean, variance,
                                    beta, gamma, 1e-8)
    y_pred = activation(z_n)
    return y_pred


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation
    graph for the neural network
    """
    prev = x
    for i in range(len(layer_sizes)):
        n = layer_sizes[i]
        activation = activations[i]
        if i == len(layer_sizes) - 1:
            layer = create_layer(prev, n, activation)
        else:
            layer = create_batch_norm_layer(prev, n, activation)
        prev = layer
    return layer


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    """
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, epsilon=epsilon,
                                       beta1=beta1, beta2=beta2)
    train = optimizer.minimize(loss)
    return train


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates learning rate operation
    """
    learning_rate = tf.train.inverse_time_decay(alpha, global_step,
                                                decay_step, decay_rate,
                                                staircase=True)
    return learning_rate


def get_batches(a, batch_size):
    """
    Divide data in batches
    """
    b_list = []
    i = 0
    m = a.shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)
    for b in range(batches):
        if b != batches-1:
            b_list.append(a[i:(i+batch_size)])
        else:
            b_list.append(a[i:])
        i += batch_size
    return b_list


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    i = np.arange(X.shape[0])
    new_i = np.random.permutation(i)
    return X[new_i], Y[new_i]


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    adam optimization, mini-batch, gradient descent
    learning rate decay, batch normalization
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    x, y = create_placeholders(nx, classes)

    y_pred = forward_prop(x, layers, activations)

    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 10)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    for e in range(epochs + 1):
        x_t, y_t = shuffle_data(X_train, Y_train)
        loss_t, acc_t = sess.run((loss, accuracy),
                                 feed_dict={x: X_train, y: Y_train})
        loss_v, acc_v = sess.run((loss, accuracy),
                                 feed_dict={x: X_valid, y: Y_valid})
        print('After {} epochs:'.format(e))
        print('\tTraining Cost: {}'.format(loss_t))
        print('\tTraining Accuracy: {}'.format(acc_t))
        print('\tValidation Cost: {}'.format(loss_v))
        print('\tValidation Accuracy: {}'.format(acc_v))

        if e < epochs:
            X_batch_t = get_batches(x_t, batch_size)
            Y_batch_t = get_batches(y_t, batch_size)
            for b in range(1, len(X_batch_t) + 1):
                sess.run(train_op, feed_dict={x: X_batch_t[b - 1],
                                              y: Y_batch_t[b - 1]})
                loss_t, acc_t = sess.run((loss, accuracy),
                                         feed_dict={x: X_batch_t[b - 1],
                                                    y: Y_batch_t[b - 1]})
                if not b % 100:
                    print('\tStep {}:'.format(b))
                    print('\t\tCost: {}'.format(loss_t))
                    print('\t\tAccuracy: {}'.format(acc_t))

    save_path = saver.save(sess, save_path)
    return save_path
