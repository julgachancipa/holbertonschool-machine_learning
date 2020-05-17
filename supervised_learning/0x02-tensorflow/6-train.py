#!/usr/bin/env python3
"""
Train
"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier
    """
    nx = X_train.shape[1]
    classes = layer_sizes[-1]
    x, y = create_placeholders(nx, classes)

    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(iterations + 1):
        loss_t, acc_t = sess.run((loss, accuracy),
                                 feed_dict={x: X_train, y: Y_train})
        loss_v, acc_v = sess.run((loss, accuracy),
                                 feed_dict={x: X_valid, y: Y_valid})
        if not i % 100:
            print('After {} iterations:'.format(i))
            print('\tTraining Cost: {}'.format(loss_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(loss_v))
            print('\tValidation Accuracy: {}'.format(acc_v))
        sess.run(train_op, feed_dict={x: X_train, y: Y_train})

    tf.train.export_meta_graph(filename=save_path,
                               collection_list=["x", "y", "y_pred"
                                                "accuracy", "train_op"])
    return save_path
