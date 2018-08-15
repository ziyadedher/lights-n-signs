"""Haar model representation.

This module contains the prediction model that will be generated by the Haar
training.
"""
from typing import List

import numpy as np

from haar.common import config
from haar.common.model import Model, PredictedObject2D, Bounds2D

cv2 = config.get_opencv_bindings()


class HaarModel(Model):
    """Bounding-box prediction model utilizing Haar cascades."""

    scale_factor: float
    min_neighbours: int

    __cascade: cv2.CascadeClassifier
    __classes: str

    def __init__(self, cascade: cv2.CascadeClassifier, classes: str) -> None:
        """Initialize a Haar cascade model.

        Contains a <cascade> and a <classes> which represents the classes the
        cascade is made to detect.
        """
        # Set default scale factor and min neighbours
        self.scale_factor = 1.1
        self.min_neighbours = 3

        self.__cascade = cascade
        self.__classes = classes

    def predict(self, image: np.ndarray) -> List[PredictedObject2D]:
        """Predict the required bounding boxes on the given <image>."""
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        predictions = self.__cascade.detectMultiScale(grayscale,
                                                      self.scale_factor,
                                                      self.min_neighbours)

        return [
            PredictedObject2D(Bounds2D(*prediction), self.__classes)
            for prediction in predictions
        ]
