"""LISA data preprocessor."""

import os
import csv

from lns.common.dataset import Dataset
from lns.common.structs import Object2D, Bounds2D
from lns.common.preprocess import Preprocessor


LIGHT_DATASET_NAME = "LISA"
SIGN_DATASET_NAME = "lisa_signs"


@Preprocessor.register_dataset_preprocessor(LIGHT_DATASET_NAME)
def _lisa(path: str) -> Dataset:  # pylint:disable=too-many-locals,too-many-branches
    images: Dataset.Images = []
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}

    day_train_path = os.path.join(path, "dayTrain")
    if not os.path.isdir(day_train_path):
        raise FileNotFoundError("Could not find `dayTrain` in LISA dataset.")

    for file_name in os.listdir(day_train_path):
        if not file_name.startswith("dayClip"):
            continue

        clip_path = os.path.join(day_train_path, file_name)
        frames_path = os.path.join(clip_path, "frames")
        annotations_path = os.path.join(clip_path, "frameAnnotationsBOX.csv")
        if not os.path.exists(frames_path):
            raise FileNotFoundError(f"Could not find frames folder {frames_path}.")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Could not find annotations file {annotations_path}")

        # Read annotations
        with open(annotations_path, "r") as annotations_file:
            reader = csv.reader(annotations_file, delimiter=";")
            for i, row in enumerate(reader):
                # Skip the first row, it is just headers
                if i == 0:
                    continue

                image_name = row[0].split("/")[-1]
                image_path = os.path.join(frames_path, image_name)

                detection_class = row[1]

                # Calculate the position and dimensions of the bounding box
                x_min = int(row[2])  # x-coordinate of top left corner
                y_min = int(row[3])  # y-coordinate of top left corner
                x_max = int(row[4])  # x-coordinate of bottom right corner
                y_max = int(row[5])  # y-coordinate of bottom right corner

                # Get the class index if it has already been registered
                # otherwise register it and select the index
                try:
                    class_index = classes.index(detection_class)
                except ValueError:
                    class_index = len(classes)
                    classes.append(detection_class)

                # Package the detection
                images.append(image_path)
                if image_path not in annotations:
                    annotations[image_path] = []
                annotations[image_path].append(
                    Object2D(Bounds2D(x_min, y_min, x_max - x_min, y_max - y_min), class_index))

    return Dataset(LIGHT_DATASET_NAME, images, classes, annotations)


@Preprocessor.register_dataset_preprocessor(SIGN_DATASET_NAME)
def _lisa_signs(path: str) -> Dataset:  # pylint: disable=too-many-locals
    images: Dataset.Images = []
    classes: Dataset.Classes = []
    annotations: Dataset.Annotations = {}

    annotations_path = os.path.join(path, "allAnnotations.csv")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError("Could not find annotations file {annotations_path} in LISA Signs dataset.")

    # Read annotations
    with open(annotations_path, "r") as annotations_file:
        reader = csv.reader(annotations_file, delimiter=";")
        for i, row in enumerate(reader):
            # Skip the first row, it is just headers
            if i == 0:
                continue

            image_name = row[0]
            image_path = os.path.join(path, image_name)

            detection_class = row[1]

            # Calculate the position and dimensions of the bounding box
            x_min = int(row[2])  # x-coordinate of top left corner
            y_min = int(row[3])  # y-coordinate of top left corner
            x_max = int(row[4])  # x-coordinate of bottom right corner
            y_max = int(row[5])  # y-coordinate of bottom right corner

            # Get the class index if it has already been registered
            # otherwise register it and select the index
            try:
                class_index = classes.index(detection_class)
            except ValueError:
                class_index = len(classes)
                classes.append(detection_class)

            # Package the detection
            images.append(image_path)
            if image_path not in annotations:
                annotations[image_path] = []
            annotations[image_path].append(Object2D(Bounds2D(x_min, y_min, x_max - x_min, y_max - y_min), class_index))

    return Dataset(SIGN_DATASET_NAME, images, classes, annotations)
