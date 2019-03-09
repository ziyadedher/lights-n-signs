import os
import yaml  # XXX: this could be sped up by using PyYaml C-bindings

from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor


DATASET_NAME = "Bosch"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _bosch(path: str) -> Dataset:
    """Preprocess and generate data for a Bosch dataset at the given path.

    Raises `FileNotFoundError` if any of the required Bosch files or
    folders is not found.
    """
    images: Dataset.Images = {DATASET_NAME: []}
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

            # Get the class index if it has already been registered
            # otherwise register it and select the index
            try:
                class_index = classes.index(label)
            except ValueError:
                class_index = len(classes)
                classes.append(label)

            # Package the detection
            if image_path not in annotations:
                annotations[image_path] = []
                images[DATASET_NAME].append(image_path)
            annotations[image_path].append({
                "class": class_index,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })

    return Dataset(DATASET_NAME, images, classes, annotations)
