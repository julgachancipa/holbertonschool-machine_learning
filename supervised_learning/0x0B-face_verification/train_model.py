#!/usr/bin/env python3
"""
Initialize Train Model
"""
import tensorflow.keras as K
import tensorflow as tf
from triplet_loss import TripletLoss


class TrainModel():
    def __init__(self, model_path, alpha):
        """
        class constructor
        :param model_path: path to the base face verification embedding model
        :param alpha: alpha to use for the triplet loss calculation
        """
        with K.utils.CustomObjectScope({'tf': tf}):
            model = K.models.load_model(model_path)
        model.summary()
