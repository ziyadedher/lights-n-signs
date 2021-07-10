"""Bosch data preprocessor."""

import os
import yaml  # XXX: this could be sped up by using PyYaml C-bindings (LibYAML)

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor


DATASET_NAME = "Bosch"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _bosch(path: str) -> Dataset:  # pylint:disable=too-many-locals
    images: Dataset.Images = []
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}

    annotations_path = os.path.join(path, "train.yaml")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(f"Could not find annotations file {annotations_path}.")
    with open(annotations_path, "r") as file:
        raw_annotations = yaml.load(file)

    for annotation in raw_annotations:
        detections = annotation["boxes"]
        image_path = os.path.abspath(os.path.join(path, annotation["path"]))

        for detection in detections:
            label = detection["label"]
            x_min = round(detection["x_min"])
            x_max = round(detection["x_max"])
            y_min = round(detection["y_min"])
            y_max = round(detection["y_max"])

            # Get the class index if it has already been registered otherwise register it and select the index
            try:
                class_index = classes.index(label)
            except ValueError:
                class_index = len(classes)
                classes.append(label)

            # Package the detection
            if image_path not in annotations:
                annotations[image_path] = []
                images.append(image_path)
            annotations[image_path].append(
                Object2D(Bounds2D(x_min, y_min, x_max - x_min, y_max - y_min), class_index))

    return Dataset(DATASET_NAME, images, classes, annotations)
