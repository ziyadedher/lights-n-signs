"""Abstract model representation.

This module consists of abstract and container classes that provide a
common interface for every technique for light and sign detection.
"""
from typing import TypeVar, List, Optional

import numpy as np  # type: ignore


class Bounds2D:
    """Two-dimensional bounds object.

    Represents two-dimensional bounding box coordinates.
    """
    __left: float
    __top: float
    __width: float
    __height: float

    def __init__(self, left: float, top: float, width: float, height: float) -> None:
        """Initialize a two-dimensional bounds object."""
        self.__left = left
        self.__top = top
        self.__width = width
        self.__height = height

    @property
    def left(self) -> float:
        """Get the left x-coordinate of this box."""
        return self.__left

    @property
    def top(self) -> float:
        """Get the top y-coordinate of this box."""
        return self.__top

    @property
    def width(self) -> float:
        """Get the width of this box."""
        return self.__width

    @property
    def height(self) -> float:
        """Get the height of this box."""
        return self.__height

    @property
    def right(self) -> float:
        """Get the right x-coordinate of this box."""
        return self.left + self.width

    @property
    def bottom(self) -> float:
        """Get the bottom y-coordinate of this box."""
        return self.top + self.height

    @property
    def area(self) -> float:
        """Get the area of this box."""
        return self.width * self.height

    def iou(self, other_box: 'Bounds2D') -> float:
        """Calculate the intersection-over-union of this box and another."""
        x_left = max(self.left, other_box.left)
        y_top = max(self.top, other_box.top)
        x_right = min(self.right, other_box.right)
        y_bottom = min(self.bottom, other_box.bottom)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        return intersection_area / float(self.area + other_box.area - intersection_area)


class PredictedObject2D:
    """Two-dimensional predicted object.

    Consists of a two-dimensional bounding box and the classes of the object
    predicted to be contained within that bounding box.
    """
    bounding_box: Bounds2D
    predicted_classes: List[str]
    scores: List[float]

    def __init__(self, bounding_box: Bounds2D, predicted_classes: List[str],
                 scores: Optional[List[float]] = None) -> None:
        """Initialize a two-dimensional predicted object."""
        self.bounding_box = bounding_box
        self.predicted_classes = predicted_classes
        self.scores = scores if scores else [1.0] * len(predicted_classes)


class Model:
    """Abstract bounding-box prediction model."""

    def predict(self, image: np.ndarray) -> List[PredictedObject2D]:
        """Predict the required bounding boxes on the given <image>."""
        raise NotImplementedError


ModelType = TypeVar("ModelType", bound=Model)
