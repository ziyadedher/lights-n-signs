"""Mocked dataset preprocessor.

Preprocesses a spoofed dataset which is assumed to be in the following format:
    mocked/
        class_1/
            img1
            img2
            ...
        class_2/
            img1
            img2
            ...
        ...

`imgX` files do not need to actually be images.
Annotations are randomly generated.
"""

import os

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor


DATASET_NAME = "mocked"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _mocked(path: str) -> Dataset:
    images: Dataset.Images = []
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}

    for i, class_name in enumerate(os.listdir(path)):
        classes.append(class_name)
        class_folder = os.path.join(path, class_name)
        for file in os.listdir(class_folder):
            image_path = os.path.join(class_folder, file)
            images.append(image_path)
            annotations[image_path] = [Object2D(Bounds2D(0, 0, 0, 0), i)]

    return Dataset(DATASET_NAME, images, classes, annotations)
