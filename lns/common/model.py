"""Abstract model representation.

This module consists of abstract and container classes that provide a
common interface for every technique for light and sign detection.
"""
from typing import TypeVar, Generic, List

import cv2          # type: ignore
import numpy as np  # type: ignore

from lns.common.structs import Object2D
from lns.common.settings import SettingsType


class Model(Generic[SettingsType]):
    """Abstract bounding-box prediction model."""

    __settings: SettingsType

    def __init__(self, settings: SettingsType) -> None:
        """Initialize a mode with the given <settings>."""
        self.__settings = settings

    @property
    def settings(self) -> SettingsType:
        """Return the settings associated with this model."""
        return self.__settings

    def predict(self, image: np.ndarray) -> List[Object2D]:
        """Predict objects on the given <image> using this model."""
        raise NotImplementedError

    def predict_path(self, path: str) -> List[Object2D]:
        """Predicts objects on the image at the given <path> using this model."""
        return self.predict(cv2.imread(path))


ModelType = TypeVar("ModelType", bound=Model)
