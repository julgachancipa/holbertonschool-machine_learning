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
    # batches = int(m / batch_size) + (m % batch_size > 0)
    batches = int(m / batch_size)
    for b in range(batches):
        """if b != batches-1:
            b_list.append(a[i:(i+32)])
        else:
            b_list.append(a[i:])"""
        b_list.append(a[i:(i + 32)])
        i += 32
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
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        for e in range(epochs):
            x_t, y_t, loss_t, acc_t = sess.run((x, y, loss, accuracy),
                                               feed_dict={x: X_train,
                                                          y: Y_train})
            loss_v, acc_v = sess.run((loss, accuracy),
                                     feed_dict={x: X_valid, y: Y_valid})
            shuffle_data(x_t, y_t)
            print('After {} epochs:'.format(e))
            print('\tTraining Cost: {}'.format(loss_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(loss_v))
            print('\tValidation Accuracy: {}'.format(acc_v))

            X_batch_t = get_batches(x_t, batch_size)
            Y_batch_t = get_batches(y_t, batch_size)
            for b in range(len(X_batch_t)):
                loss_t, acc_t = sess.run((loss, accuracy),
                                         feed_dict={x: X_batch_t[b],
                                                    y: Y_batch_t[b]})
                if not b % 100 and b > 0:
                    print('\tStep {}:'.format(b))
                    print('\t\tCost: {}'.format(loss_t))
                    print('\t\tAccuracy: {}'.format(acc_t))
                sess.run(train_op, feed_dict={x: X_batch_t[b],
                                              y: Y_batch_t[b]})
