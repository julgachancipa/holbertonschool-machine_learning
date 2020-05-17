#!/usr/bin/env python3
"""
Evaluate
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(save_path))
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        y_pred_e, loss_e, acc_e = sess.run((y_pred, loss, accuracy),
                                           feed_dict={x: X, y: Y})
    return y_pred_e, acc_e, loss_e
