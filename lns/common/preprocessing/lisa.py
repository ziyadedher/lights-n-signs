import os
import csv

from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor


DATASET_NAME = "LISA"


@Preprocessor.register_dataset_preprocessor(DATASET_NAME)
def _lisa(path: str) -> Dataset:
    """Preprocess and generate data for a LISA dataset at the given path.

    Only uses the `dayTrain` data subset.
    Raises `FileNotFoundError` if any of the required LISA files or folders
    is not found.
    """
    images: Dataset.Images = {DATASET_NAME: []}
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
                x_min = int(row[2])      # x-coordinate of top left corner
                y_min = int(row[3])      # y-coordinate of top left corner
                x_max = int(row[4])      # x-coordinate of bottom right corner
                y_max = int(row[5])      # y-coordinate of bottom right corner

                # Get the class index if it has already been registered
                # otherwise register it and select the index
                try:
                    class_index = classes.index(detection_class)
                except ValueError:
                    class_index = len(classes)
                    classes.append(detection_class)

                # Package the detection
                images[DATASET_NAME].append(image_path)
                if image_path not in annotations:
                    annotations[image_path] = []
                annotations[image_path].append({
                    "class": class_index,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                })

    return Dataset(DATASET_NAME, images, classes, annotations)
