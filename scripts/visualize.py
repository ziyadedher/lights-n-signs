from typing import Optional

import sys

import cv2  # type: ignore

from lns.common.model import Model
from lns.common.dataset import Dataset
# from lns.common.utils.visualization import put_labels_on_image, put_predictions_on_image


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
 #       image = put_predictions_on_image(image, model.predict(image))

    cv2.imshow("visualization", image)
    key = cv2.waitKey(0)
    while key != ord("a"):
        key = cv2.waitKey(0)
        if key == 27:
            sys.exit(0)

def put_labels_on_image(image: np.ndarray, labels: Dataset.Labels):
    for label in labels:
        image = cv2.rectangle(image, (label.left, label.top), (label.right, label.bottom), (0, 255, 0))
    return image

if __name__ == '__main__':
    cv2.namedWindow("visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("visualization", 1920, 1080)

    from lns.common.preprocess import Preprocessor
    bosch = Preprocessor.preprocess("Bosch")
    #lights = Preprocessor.preprocess("lights")
    #scale = Preprocessor.preprocess("scale_lights")
    # print(Preprocessor._dataset_preprocessors)
    # mturk = Preprocessor.preprocess("scale_lights")

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

    #from lns.squeezedet.model import SqueezeDetModel
    #model = SqueezeDetModel("/home/lns/lns/xiyan/models/alllights-414000/train/model.ckpt-415500")

    annotations = dataset.annotations
    for image_path in annotations:
        visualize_image(image_path, model=None, visualize_model=False, labels=annotations[image_path], show_labels=True)
