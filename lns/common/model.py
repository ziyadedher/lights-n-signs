"""Abstract model representation.

This module consists of abstract and container classes that provide a
common interface for every technique for light and sign detection.
"""
from typing import TypeVar, List

import numpy as np  # type: ignore

from lns.common.structs import Object2D


class Model:
    """Abstract bounding-box prediction model."""

    def predict(self, image: np.ndarray) -> List[Object2D]:
        """Predict objects on the given <image> using this model."""
        raise NotImplementedError


ModelType = TypeVar("ModelType", bound=Model)
