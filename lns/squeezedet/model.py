"""SqueezeDet model representation.

This module contains the prediction model that will be generated by the
SqueezeDet training.
"""
from typing import List

import cv2  # type: ignore
import numpy as np  # type: ignore

from lns.common.model import Model
from lns.common.structs import Bounds2D, Object2D
from lns.squeezedet._lib.config.create_config import load_dict
from lns.squeezedet._lib.model.evaluation import filter_batch
from lns.squeezedet._lib.model.squeezeDet import SqueezeDet
from lns.squeezedet.settings import SqueezedetSettings


class SqueezedetModel(Model):
    """Bounding-box prediction model utilizing SqueezeDet."""

    __squeeze: SqueezeDet

    def __init__(self, config_file: str, settings: SqueezedetSettings) -> None:
        """Initialize a SqueezeDet model with the given model and config."""
        super().__init__(settings)
        self.__squeeze = SqueezeDet(load_dict(config_file))

    def predict(self, image: np.ndarray) -> List[Object2D]:
        """Predict the required bounding boxes on the given <image>."""
        # TODO: resize / other manipulations
        boxes, classes, scores = filter_batch(self.__squeeze.model.predict([image]), self.__squeeze.config)

        print(boxes, classes, scores)
        return [Object2D(Bounds2D(0, 0, 0, 0), class_index=0, score=0)]
