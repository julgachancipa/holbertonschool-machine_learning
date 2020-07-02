#!/usr/bin/env python3
"""
This file contain the Yolo class
"""
import tensorflow.keras as K


class Yolo():
    """
    This is a class to use the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Everything begins here """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            data = f.read()
        self.class_names = data.split()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
