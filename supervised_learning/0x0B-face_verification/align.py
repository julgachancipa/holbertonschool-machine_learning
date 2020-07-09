#!/usr/bin/env python3
"""
Initialize Face Align
"""
import dlib
import cv2
import numpy as np
from imutils import face_utils, resize


class FaceAlign():

    def __init__(self, shape_predictor_path):
        """
        class constructor
        :param self:
        :param shape_predictor_path: path to the dlib shape predictor model
        :return: Nothing
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        Detects a face in an image

        :param image: numpy.ndarray of rank 3 containing an image from which
        to detect a face

        :return: dlib.rectangle containing the boundary box for the face in
        the image, or None on failure
        """
        rects = self.detector(image)

        if not len(rects):
            return dlib.get_rect(image)

        max_area = [None, 0]
        for rect in rects:
            if rect.area() > max_area[1]:
                max_area = [rect, rect.area()]
        return max_area[0]

    def find_landmarks(self, image, detection):
        """
        Finds facial landmarks

        :param image: numpy.ndarray of an image from which to find facial
        landmarks
        :param detection: dlib.rectangle containing the boundary box of the
         face in the image

        :return: numpy.ndarray of shape (p, 2)containing the landmark points,
        or None on failure
        """
        shape = self.shape_predictor(image, detection)
        shape = face_utils.shape_to_np(shape)
        return shape

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        Aligns an image for face verification

        :param image: numpy.ndarray of rank 3 containing the image to be
        aligned
        :param landmark_indices: numpy.ndarray of shape (3,) containing the
        indices of the three landmark points that should be used for the
        affine transformation
        :param anchor_points: numpy.ndarray of shape (3, 2) containing the
        destination points for the affine transformation, scaled to the
        range [0, 1]
        :param size: desired size of the aligned image

        :return: numpy.ndarray of shape (size, size, 3) containing the aligned
        image, or None if no face is detected
        """
        rect = self.detect(image)
        shape = self.find_landmarks(image, rect)
        landmark = shape[landmark_indices]
        M = cv2.getAffineTransform(landmark.astype(np.float32),
                                   (anchor_points * size).astype(np.float32))

        alg = cv2.warpAffine(image, M, (size, size))
        return alg
