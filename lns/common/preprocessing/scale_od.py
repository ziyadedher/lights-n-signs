"""Scale OD pedestrian dataset preprocessor."""

import os

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor


DATASET_NAME = "SCALE"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _scale_od_ped(path: str) -> Dataset:  # noqa
    images: Dataset.Images = []
    classes: Dataset.Classes = ["Pedestrian"]
    annotations: Dataset.Annotations = {}

    images_path = os.path.join(path, "training", "image_2")
    labels_path = os.path.join(path, "training", "label_pedonly")

    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path, image_name)
        base_name = image_name.split(".")[0]

        label_path = os.path.join(labels_path, base_name + ".txt")
        if os.path.exists(label_path):
            images.append(image_path)
            image_annotations = []
            with open(label_path, "r") as file:
                for line in file:
                    left, top, right, bottom = [float(s) for s in line.split()[4:8]]
                    width, height = right - left, bottom - top
                    bounds = Bounds2D(left, top, width, height)
                    image_annotations.append(Object2D(bounds, 0))
            annotations[image_path] = image_annotations

    return Dataset(DATASET_NAME, images, classes, annotations)
