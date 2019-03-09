from typing import List

import cv2  # type: ignore
import numpy as np  # type: ignore

from lns.common.model import PredictedObject2D
from lns.common.dataset import Dataset

STROKE_WIDTH = 1
COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "off": (0, 0, 0),
}
TEXT_OFFSET = (0, 15)
TEXT_SIZE = 0.5


def put_labels_on_image(image: np.ndarray, labels: Dataset.Labels) -> np.ndarray:
    for label in labels:
        image = cv2.rectangle(
            image, (int(label["x_min"]), int(label["y_min"])), (int(label["x_max"]), int(label["y_max"])),
            (255, 255, 255), STROKE_WIDTH
        )
    return image


def put_predictions_on_image(image: np.ndarray, predictions: List[PredictedObject2D]) -> np.ndarray:
    for prediction in predictions:
        predicted_class = prediction.predicted_classes[0]
        confidence = prediction.scores[0]
        box = prediction.bounding_box
        color = COLOR_MAP.get(predicted_class, (255, 255, 255))

        cv2.rectangle(
            image, (int(box.left), int(box.top)), (int(box.right), int(box.bottom)),
            color, STROKE_WIDTH
        )
        cv2.putText(
            image, f"{predicted_class}:{confidence:.2f}",
            (int(box.left + TEXT_OFFSET[0]), int(box.bottom + TEXT_OFFSET[1])),
            cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, color
        )
    return image
