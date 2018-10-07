"""Test Model

This module contains a randomized prediction model used for testing the 
benchmarking module
"""

from typing import List, Tuple
from random import randint

import cv2          # type: ignore
import numpy as np  # type: ignore

from common.model import Model, PredictedObject2D, Bounds2D


class TestModel(Model):
    """Randomized bounding-box prediction model"""

    __classes: List[str]

    def __init__(self, classes: List[str]) -> None:
        """Initialize the randomized model""" 

        self.__classes = classes

    def predict(self, image: np.ndarray) -> List[PredictedObject2D]:
        """Predict the required bounding boxes on the given <image>."""
        predictions: List[Tuple[int, int, int, int]] = \
            [[randint(0, i) for i in [image.shape[0], image.shape[1], 40, 60]]]

        return [
            PredictedObject2D(Bounds2D(*prediction), self.__classes)
            for prediction in predictions
        ]
