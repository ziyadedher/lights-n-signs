"""Manages internal preprocessing methods.

Contains dedicated preprocessing functions for each dataset and the
preprocessing data structure.
"""
from typing import Dict, List

import os
import csv
import json
import copy
import yaml  # XXX: this could be sped up by using PyYaml C-bindings
import random


class Dataset:
    """Read-only container structure for data generated by preprocessing."""

    _name: str

    __images: Dict[str, List[str]]
    __classes: List[str]
    __annotations: Dict[str, List[Dict[str, int]]]

    def __init__(self, name: str,
                 images: Dict[str, List[str]], classes: List[str],
                 annotations: Dict[str, List[Dict[str, int]]]) -> None:
        """Initialize the data structure.

        <name> is a unique name for this dataset.
        <images> is a mapping of dataset name to list of absolute paths to the
        images in the dataset.
        <classes> is an indexed list of classes
        <annotations> is a mapping of image path to a list of "detections"
        represented by a dictionary containing keys `class` corresponding
        to the class index detected, `x_min`, `y_min` corresponding to the
        x-coordinate and y-coordinate of the top left corner of the bounding
        box, and `x_max`, `y_max` corresponding to the x-coordinate and
        y-coordinate of the bottom right corner of the bounding box.
        """
        self._name = name
        self.__images = images
        self.__classes = classes
        self.__annotations = annotations

    @property
    def name(self) -> str:
        """Get the name of this dataset."""
        return self._name

    @property
    def images(self) -> Dict[str, List[str]]:
        """Get a list of paths to all images available in the dataset."""
        return copy.deepcopy(self.__images)

    @property
    def classes(self) -> List[str]:
        """Get a mapping of ID to name for all classes in the dataset."""
        return copy.deepcopy(self.__classes)

    @property
    def annotations(self) -> Dict[str, List[Dict[str, int]]]:
        """Get all image annotations.

        Image annotations are structured as a mapping of absolute image path
        (as given in `self.images`) to a list of detections. Each detection
        consists of a mapping from detection key to its respective information.

        Available detection keys are
        `class`, `x_min`, `y_min`, `x_max`, `y_max`.
        """
        return copy.deepcopy(self.__annotations)

    def merge_classes(self, mapping: Dict[str, List[str]]) -> 'Dataset':
        """Get a new `Dataset` that has classes merged together.

        Merges the classes under the values in <mapping> under the class given
        by the respective key.
        """
        images = self.images
        classes = list(mapping.keys())
        annotations = self.annotations

        for path, annotation in annotations.items():
            for detection in annotation:
                # Change the detection class if required
                for new_class, mapping_classes in mapping.items():
                    if self.classes[detection["class"]] in mapping_classes:
                        detection["class"] = classes.index(new_class)
                        break

        return Dataset(self.name, images, classes, annotations)

    def __add__(self, other: 'Dataset') -> 'Dataset':
        """Magic method for adding two preprocessing data objects."""
        return Dataset(f"{self.name}-{other.name}",
                       {**self.images, **other.images},
                       self.classes + other.classes,
                       {**self.annotations, **other.annotations})


def preprocess_LISA(LISA_path: str) -> Dataset:
    """Preprocess and generate data for a LISA dataset at the given path.

    Only uses the `dayTrain` data subset.
    Raises `FileNotFoundError` if any of the required LISA files or folders
    is not found.
    """
    day_train_path = os.path.join(LISA_path, "dayTrain")
    if not os.path.isdir(day_train_path):
        raise FileNotFoundError("Could not find `dayTrain` in LISA dataset.")

    # Define lists containing info for test + train structures
    images_train: List[str] = []
    detection_classes_train: List[str] = []
    annotations_train: Dict[str, List[Dict[str, int]]] = {}

    images_test: List[str] = []
    detection_classes_test: List[str] = []
    annotations_test: Dict[str, List[Dict[str, int]]] = {}

    # Generate the random seed
    random.seed(1)  # Can be arbitrary const

    for file_name in os.listdir(day_train_path):
        if not file_name.startswith("dayClip"):
            continue

        clip_path = os.path.join(day_train_path, file_name)
        frames_path = os.path.join(clip_path, "frames")
        annotations_path = os.path.join(clip_path, "frameAnnotationsBOX.csv")
        if not os.path.exists(frames_path):
            raise FileNotFoundError(
                f"Could not find frames folder {frames_path}."
            )
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(
                f"Could not find annotations file {annotations_path}"
            )

        # Register all the images
        for image_name in os.listdir(frames_path):
            if random.random() < 0.1:
                images_test.append(os.path.join(frames_path, image_name))
            else:
                images_train.append(os.path.join(frames_path, image_name))

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

                if image_path in images_test:
                    try:
                        class_index = detection_classes_test.index(
                            detection_class
                        )
                    except ValueError:
                        class_index = len(detection_classes_test)
                        detection_classes_test.append(detection_class)

                    # Package the detection
                    if image_path not in annotations_test:
                        annotations_test[image_path] = []
                        annotations_test[image_path].append({
                            "class": class_index,
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        })
                else:
                    try:
                        class_index = detection_classes_train.index(
                            detection_class
                        )
                    except ValueError:
                        class_index = len(detection_classes_train)
                        detection_classes_train.append(detection_class)

                    # Package the detection
                    if image_path not in annotations_train:
                        annotations_train[image_path] = []
                        annotations_train[image_path].append({
                            "class": class_index,
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        })

    # save the test_struct in file for future reference
    with open("test_data.json", "w") as rs:
        json.dump({
            "images": images_test,
            "classes": detection_classes_test,
            "annotations": annotations_test
        }, rs)

    train_struct = Dataset("LISA", {"LISA": images_train},
                           detection_classes_train, annotations_train)
    return train_struct


def preprocess_bosch(bosch_path: str) -> Dataset:
    """Preprocess and generate data for a Bosch dataset at the given path.

    Raises `FileNotFoundError` if any of the required Bosch files or
    folders is not found.
    """
    annotations_path = os.path.join(bosch_path, "train.yaml")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(
            f"Could not find annotations file {annotations_path}."
        )
    with open(annotations_path, "r") as file:
        raw_annotations = yaml.load(file)

    images: List[str] = []
    detection_classes: List[str] = []
    annotations: Dict[str, List[Dict[str, int]]] = {}

    for annotation in raw_annotations:
        detections = annotation["boxes"]
        image_path = os.path.abspath(
            os.path.join(bosch_path, annotation["path"])
        )
        images.append(image_path)

        for detection in detections:
            label = detection["label"]
            x_min = round(detection["x_min"])
            x_max = round(detection["x_max"])
            y_min = round(detection["y_min"])
            y_max = round(detection["y_max"])

            # Get the class index if it has already been registered
            # otherwise register it and select the index
            try:
                class_index = detection_classes.index(label)
            except ValueError:
                class_index = len(detection_classes)
                detection_classes.append(label)

            # Package the detection
            if image_path not in annotations:
                annotations[image_path] = []
            annotations[image_path].append({
                "class": class_index,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })

    return Dataset("Bosch", {"Bosch": images}, detection_classes, annotations)
