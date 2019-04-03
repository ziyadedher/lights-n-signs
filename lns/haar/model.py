"""Haar model representation.

This module contains the prediction model that will be generated by the Haar
training.
"""
from typing import List, Tuple

import os
import cv2          # type: ignore
import numpy as np  # type: ignore

from lns_common.model import Model, PredictedObject2D, Bounds2D


class HaarModel(Model):
    """Bounding-box prediction model utilizing Haar cascades."""

    scale_factor: float
    min_neighbours: int

    __cascade: cv2.CascadeClassifier
    __classes: List[str]

    def __init__(self, cascade_file: str,
                 classes: List[str]) -> None:
        """Initialize a Haar cascade model.

        Contains a <cascade> and a <classes> which represents the classes the
        cascade is made to detect.
        """
        # Set default scale factor and min neighbours
        self.scale_factor = 1.1
        self.min_neighbours = 3

        cascade_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.expanduser(cascade_file))
        self.__cascade = cv2.CascadeClassifier(cascade_file)
        self.__classes = classes

    def predict(self, image: np.ndarray) -> List[PredictedObject2D]:
        """Predict the required bounding boxes on the given <image>."""
        grayscale: np.ndarray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        predictions: List[Tuple[int, int, int, int]] = \
            self.__cascade.detectMultiScale(
                grayscale, self.scale_factor, self.min_neighbours
        )

        return [
            PredictedObject2D(Bounds2D(*prediction), self.__classes)
            for prediction in predictions
        ]