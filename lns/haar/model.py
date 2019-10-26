"""Haar model representation.

This module contains the prediction model that will be generated by the Haar
training.
"""
from typing import Optional, List, Tuple

import cv2          # type: ignore
import numpy as np  # type: ignore

from lns.common.model import Model
from lns.common.structs import Object2D, Bounds2D
from lns.haar.settings import HaarSettings


class HaarModel(Model[HaarSettings]):
    """Bounding-box prediction model utilizing Haar cascades."""

    __cascade: cv2.CascadeClassifier

    def __init__(self, cascade_file: str, settings: Optional[HaarSettings] = None) -> None:
        """Initialize a Haar cascade model."""
        if not settings:
            settings = HaarSettings()

        super().__init__(settings)

        self.__cascade = cv2.CascadeClassifier(cascade_file)

    def predict(self, image: np.ndarray) -> List[Object2D]:
        """Predict the required bounding boxes on the given <image>."""
        predictions: List[Tuple[int, int, int, int]] = self.__cascade.detectMultiScale(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), self.settings.scale_factor, self.settings.min_neighbours)
        return [Object2D(Bounds2D(*prediction), self.settings.class_index) for prediction in predictions]
