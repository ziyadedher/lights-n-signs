from typing import Optional

import sys

import cv2  # type: ignore

from lns.common.model import Model
from lns.common.dataset import Dataset
from lns.common.utils.visualization import put_labels_on_image, put_predictions_on_image


cv2.namedWindow("visualization", cv2.WINDOW_NORMAL)
cv2.resizeWindow("visualization", 1920, 1080)


def visualize_image(image_path: str, *,
                    model: Optional[Model] = None, visualize_model: bool = False,
                    labels: Optional[Dataset.Labels] = None, show_labels: bool = False) -> None:
    image = cv2.imread(image_path)

    if show_labels:
        if labels is None:
            raise ValueError("Labels cannot be none if <show_labels> is set to `True`.")
        image = put_labels_on_image(image, labels)

    if visualize_model:
        if model is None:
            raise ValueError("Need to set a model if <visualize_model> is set to `True`.")
        image = put_predictions_on_image(image, model.predict(image))

    cv2.imshow("visualization", image)
    key = cv2.waitKey(0)
    while key != ord("a"):
        key = cv2.waitKey(0)
        if key == 27:
            sys.exit(0)


if __name__ == '__main__':
    from lns.common.preprocess import Preprocessor
    #bosch = Preprocessor.preprocess("Bosch")
    #lights = Preprocessor.preprocess("lights")
    #scale = Preprocessor.preprocess("scale_lights")
    print(Preprocessor._dataset_preprocessors)
    mturk = Preprocessor.preprocess("mturk")
    dataset = mturk
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
    dataset = dataset.minimum_area(0)

    from lns.squeezedet.model import SqueezeDetModel
    model = SqueezeDetModel("/home/lns/lns/xiyan/models/alllights-414000/train/model.ckpt-415500")

    annotations = dataset.annotations
    for image_paths in dataset.image_split(0.1)[0].values():
        for image_path in image_paths:
            visualize_image(image_path, model=model, visualize_model=False, labels=annotations[image_path], show_labels=True)
