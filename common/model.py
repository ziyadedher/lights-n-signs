"""Abstract model representation.

This module consists of abstract and container classes that provide a
common interface for every technique for light and sign detection.
"""
from typing import List

import numpy as np  # type: ignore


class Bounds2D:
    """Two-dimensional bounds object.

    Represents two-dimensional bounding box coordinates.
    """

    left: float
    top: float
    width: float
    height: float

    def __init__(self, left: float, top: float,
                 width: float, height: float) -> None:
        """Initialize a two-dimensional bounds object."""
        self.left = left
        self.top = top
        self.width = width
        self.height = height


class PredictedObject2D:
    """Two-dimensional predicted object.

    Consists of a two-dimensional bounding box and the classes of the object
    predicted to be contained within that bounding box.
    """

    bounding_box: Bounds2D
    predicted_classes: List[str]

    def __init__(self, bounding_box: Bounds2D,
                 predicted_classes: List[str]) -> None:
        """Initialize a two-dimensional predicted object."""
        self.bounding_box = bounding_box
        self.predicted_classes = predicted_classes


class Model:
    """Abstract bounding-box prediction model."""

    def predict(self, image: np.ndarray) -> List[PredictedObject2D]:
        """Predict the required bounding boxes on the given <image>."""
        raise NotImplementedError
