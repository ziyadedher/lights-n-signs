"""Abstract model representation.

This module consists of abstract and container classes that provide a
common interface for every technique for light and sign detection.
"""
from typing import TypeVar, List

import cv2          # type: ignore
import numpy as np  # type: ignore

from lns.common.structs import Object2D


class Model:  # noqa: R903
    """Abstract bounding-box prediction model."""

    def predict(self, image: np.ndarray) -> List[Object2D]:
        """Predict objects on the given <image> using this model."""
        raise NotImplementedError

    def predict_path(self, path: str) -> List[Object2D]:
        """Predicts objects on the image at the given <path> using this model."""
        return self.predict(cv2.imread(path))


ModelType = TypeVar("ModelType", bound=Model)
