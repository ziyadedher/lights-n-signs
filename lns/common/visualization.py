""""""

from typing import Optional, Union

import random

import cv2  # type: ignore

from lns.common.model import Model
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor


QUIT_KEY = "q"


def visualize(dataset: Union[Dataset, str], model: Optional[Model] = None, *,
              shuffle: bool = False) -> None:
    """Visualize the given <dataset>.

    If a <model> is supplied, visualizes with predictions from that model.
    If <shuffle> is set to `True`, randomly shuffles the images in the
    dataset before visualizing.
    """
    thickness = 1
    color = (255, 255, 255)

    if isinstance(dataset, str):
        dataset = Preprocessor.preprocess(dataset)
    window_name = f"visualization_{dataset.name}"

    images = dataset.images
    if shuffle:
        random.shuffle(images)

    annotations = dataset.annotations
    for image_path in images:
        image = cv2.imread(image_path)

        if model:
            labels = annotations[image_path]
            for label in labels:
                cv2.rectangle(
                    image,
                    (label.bounds.left, label.bounds.top), (label.bounds.right, label.bounds.bottom),
                    color=color, thickness=thickness
                )

        cv2.imshow(window_name, image)
        if cv2.waitKey() == ord(QUIT_KEY):
            break
