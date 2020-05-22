#!/usr/bin/env python3
"""
Mini-Batch
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


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


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network
    model using mini-batch gradient descent
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(load_path))
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

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
