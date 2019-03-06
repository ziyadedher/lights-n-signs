from typing import Optional

import sys

import cv2  # type: ignore

from lns.common.model import Model
from lns.common.dataset import Dataset


STROKE_WIDTH = 1
COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "off": (0, 0, 0),
}
TEXT_OFFSET = (0, 15)
TEXT_SIZE = 0.5


def visualize_image(model: Model, image_path: str, *,
                    labels: Optional[Dataset.Labels] = None, show_labels: bool = False) -> None:
    image = cv2.imread(image_path)

    if show_labels:
        if labels is None:
            raise ValueError("Labels cannot be none if <show_labels> is set to `True`.")

        for label in labels:
            cv2.rectangle(
                image, (int(label["x_min"]), int(label["y_min"])), (int(label["x_max"]), int(label["y_max"])),
                (255, 255, 255), STROKE_WIDTH
            )

    predictions = model.predict(image)
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

    cv2.imshow("visualization", image)
    key = cv2.waitKey(0)
    if key == 27:
        sys.exit(0)


if __name__ == '__main__':
    from lns.common.preprocess import Preprocessor
    Preprocessor.register_default_preprocessors()
    # bosch = Preprocessor.preprocess("Bosch")
    lisa = Preprocessor.preprocess("LISA")
    dataset = lisa
    dataset = dataset.merge_classes({
        "green": [
            "GreenLeft", "Green", "GreenRight", "GreenStraight",
            "GreenStraightRight", "GreenStraightLeft", "Green traffic light"
        ],
        "red": [
            "Yellow", "RedLeft", "Red", "RedRight", "RedStraight",
            "RedStraightLeft", "Red traffic light", "Yellow traffic light"
        ],
        "off": ["off"]
    })

    from lns.squeezedet.model import SqueezeDetModel
    model = SqueezeDetModel("/home/lns/lns/xiyan/models/alllights-414000/train/model.ckpt-415500")

    annotations = dataset.annotations
    for image_paths in dataset.image_split(0.1)[0].values():
        for image_path in image_paths:
            visualize_image(model, image_path, labels=annotations[image_path], show_labels=True)
