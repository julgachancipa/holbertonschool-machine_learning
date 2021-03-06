#!/usr/bin/env python3
"""
This file contain the Yolo class
"""
import tensorflow.keras as K
import numpy as np


def _sigmoid(x):
    """
    Sigmoid function
    :param x: x input
    :return: activation
    """
    return 1. / (1. + np.exp(-x))


class Yolo():
    """
    This is a class to use the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor
        :param model_path: is the path to where a Darknet Keras model is stored
        :param classes_path: s the path to where the list of class names used
        for the Darknet model, listed in order of index, can be found
        :param class_t: is a float representing the box score threshold for the
        initial filtering step
        :param nms_t: is a float representing the IOU threshold for non-max
        suppression
        :param anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
            outputs is the number of outputs (predictions) made by the Darknet
            model anchor_boxes is the number of anchor boxes used for each
            prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            data = f.read()
        self.class_names = data.split()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Outputs
        :param outputs: is a list of numpy.ndarrays containing the predictions
        from the Darknet model for a single image
        :param image_size: is a numpy.ndarray containing the image’s original
        size [image_height, image_width]
        :return: tuple of (boxes, box_confidences, box_class_probs)
        """

        boxes, box_confidences, box_class_probs = [], [], []

        for i in range(len(outputs)):
            ih, iw = image_size
            t_xy, t_wh, objectness, classes = np.split(outputs[i], (2, 4, 5),
                                                       axis=-1)

            box_confidences.append(_sigmoid(objectness))
            box_class_probs.append(_sigmoid(classes))

            grid_size = np.shape(outputs[i])[1]
            # bh bw debe ser normalizado dividiendose por el input shape
            # x = bx -bw / 2
            # y = by -bh /2
            # meshgrid generates a grid that repeats by given range.
            # It's the Cx and Cy in YoloV3 paper.
            # for example, tf.meshgrid(tf.range(3), tf.range(3)) will generate
            # a list with two elements
            # note that in real code, the grid_size should be something
            # like 13,
            # 26, 52 for examples here and below
            #
            # [[0, 1, 2],
            #  [0, 1, 2],
            #  [0, 1, 2]]
            #
            # [[0, 0, 0],
            #  [1, 1, 1],
            #  [2, 2, 2]]
            #
            C_xy = np.meshgrid(range(grid_size), range(grid_size))

            # next, we stack two items in the list together in the last
            # dimension, so that we can interleve these elements together
            # and become this:
            #
            # [[[0, 0], [1, 0], [2, 0]],
            #  [[0, 1], [1, 1], [2, 1]],
            #  [[0, 2], [1, 2], [2, 2]]]
            #
            C_xy = np.stack(C_xy, axis=-1)

            # let's add an empty dimension at axis=2 to expand the tensor
            # to this:
            #
            # [[[[0, 0]], [[1, 0]], [[2, 0]]],
            #  [[[0, 1]], [[1, 1]], [[2, 1]]],
            #  [[[0, 2]], [[1, 2]], [[2, 2]]]]
            #
            # at this moment, we now have a grid, which can always give
            # us (y, x)
            # if we access grid[x][y]. For example, grid[0][1] == [[1, 0]]

            C_xy = np.expand_dims(C_xy, axis=2)  # [gx, gy, 1, 2]

            # YoloV2, YoloV3:
            # bx = sigmoid(tx) + Cx
            # by = sigmoid(ty) + Cy
            #
            # for example, if all elements in b_xy are (0.1, 0.2),
            # the result will be
            #
            # [[[[0.1, 0.2]], [[1.1, 0.2]], [[2.1, 0.2]]],
            #  [[[0.1, 1.2]], [[1.1, 1.2]], [[2.1, 1.2]]],
            #  [[[0.1, 2.2]], [[1.1, 2.2]], [[2.1, 2.2]]]]
            #
            b_xy = _sigmoid(t_xy) + C_xy

            # finally, divide this absolute box_xy by grid_size,
            # and then we will get the normalized bbox centroids
            # for each anchor in each grid cell. b_xy is now in shape
            # (batch_size, grid_size, grid_size, num_anchor, 2)
            #
            # [[[[0.1/3, 0.2/3]], [[1.1/3, 0.2/3]], [[2.1/3, 0.2/3]]],
            #  [[[0.1/3, 1.2/3]], [[1.1/3, 1.2]/3], [[2.1/3, 1.2/3]]],
            #  [[[0.1/3, 2.2/3]], [[1.1/3, 2.2/3]], [[2.1/3, 2.2/3]]]]
            #
            b_xy = b_xy / grid_size

            # YoloV2:
            # "If the cell is offset from the top left corner of the image
            # by (cx , cy)
            # and the bounding box prior has width and height pw , ph , then
            # the predictions correspond to: "
            #
            # https://github.com/pjreddie/darknet/issues/
            # 568#issuecomment-469600294
            # "It’s OK for the predicted box to be wider and/or taller than
            # the original image, but
            # it does not make sense for the box to have a negative width or
            # height. That’s why
            # we take the exponent of the predicted number."
            inp = self.model.input_shape[1:3]
            b_wh = (np.exp(t_wh) / inp) * self.anchors[i]

            bx = b_xy[:, :, :, :1]
            by = b_xy[:, :, :, 1:2]
            bw = b_wh[:, :, :, :1]
            bh = b_wh[:, :, :, 1:2]

            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]

            boxes.append(np.concatenate([x1, y1, x2, y2], axis=-1))

        return boxes, box_confidences, box_class_probs
