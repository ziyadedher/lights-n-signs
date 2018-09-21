"""Data processing for Haar cascade training.

Manages all data processing for the generation of data ready to be trained
on with OpenCV Haar training scripts.
"""
from typing import List, Dict

import os
import shutil

import cv2             # type: ignore
from tqdm import tqdm  # type: ignore

from lns_common import config
from lns_common.preprocess.preprocessing import Dataset


class HaarData:
    """Data container for all Haar processed data.

    Contains positive annotations for each type of light and negative
    annotations for each type of light as well from the dataset
    """

    __positive_annotations: Dict[str, str]
    __negative_annotations: Dict[str, str]

    def __init__(self,
                 positive_annotations: Dict[str, str],
                 negative_annotations: Dict[str, str]) -> None:
        """Initialize the data structure."""
        self.__positive_annotations = positive_annotations
        self.__negative_annotations = negative_annotations

    def get_positive_annotation(self, light_type: str) -> str:
        """Get the path to a positive annotation file for the given light type.

        Raises `KeyError` if no such light type is available.
        """
        try:
            return self.__positive_annotations[light_type]
        except KeyError as e:
            raise e

    def get_negative_annotation(self, light_type: str) -> str:
        """Get the path to a negative annotation file for the given light type.

        Raises `KeyError` if no such light type is available.
        """
        try:
            return self.__negative_annotations[light_type]
        except KeyError as e:
            raise e


class HaarProcessor:
    """Haar processor responsible for data processing to Haar-valid formats."""

    BASE_DATA_FOLDER = os.path.join(config.RESOURCES_ROOT, "haar/data")

    @classmethod
    def process(cls, dataset: Dataset, force: bool = False) -> HaarData:
        """Process all required data from the dataset with the given name.

        Setting <force> to `True` will force a processing even if the images
        already exist on file.

        Raises `NoPreprocessorException` if a preprocessor for the dataset does
        not exist.
        """
        # TODO: structure this function better

        # Register all folders
        data_folder = os.path.join(cls.BASE_DATA_FOLDER, dataset.name)
        annotations_folder = os.path.join(data_folder, "annotations")
        images_folder = os.path.join(data_folder, "images")

        # Create base data folder if it does not exist
        if not os.path.exists(cls.BASE_DATA_FOLDER):
            os.makedirs(cls.BASE_DATA_FOLDER)
        # Create required folders if they do not exist
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        # Remove annotations folder to be regenerated
        if os.path.exists(annotations_folder):
            shutil.rmtree(annotations_folder)
        os.makedirs(annotations_folder)

        # Open the positive and negative annotation files
        positive_annotations_files = {
            class_name: open(os.path.join(
                annotations_folder, class_name + "_positive"
            ), "w") for class_name in dataset.classes
        }
        negative_annotations_files = {
            class_name: open(os.path.join(
                annotations_folder, class_name + "_negative"
            ), "w") for class_name in dataset.classes
        }

        # Set up for reading annotations
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enumeration = enumerate(dataset.annotations.items())

        # Read all annotations
        with tqdm(desc="Preprocessing",
                  total=len(dataset.annotations.keys()),
                  miniters=1) as bar:
            for i, (image_path, labels) in enumeration:
                # Update the progress bar
                bar.update()

                # Create gray images
                new_image_path = os.path.abspath(os.path.join(
                    images_folder, f"{i}.png"
                ))
                # Skip image creation if force is not set to True and
                # the image already exists
                if not force and not os.path.exists(new_image_path):
                    cv2.imwrite(new_image_path,
                                clahe.apply(cv2.imread(image_path, 0)))

                # Get relative path to image
                image_relative = os.path.relpath(
                    os.path.join(images_folder, f"{i}.png"),
                    start=annotations_folder
                )

                # Store the annotations in a way easier to represent for Haar
                light_detections: Dict[str, List[List[int]]] = {}

                # Go through each detection and populate the above dictionary
                for label in labels:
                    class_name = dataset.classes[label["class"]]
                    x_min = label["x_min"]
                    y_min = label["y_min"]
                    width = label["x_max"] - x_min
                    height = label["y_max"] - y_min

                    if class_name not in light_detections:
                        light_detections[class_name] = []
                    light_detections[class_name].append(
                        [x_min, y_min, width, height]
                    )

                # Append to the positive annotations file
                for light_type, detections in light_detections.items():
                    detections_string = " ".join(
                        " ".join(str(item) for item in detection)
                        for detection in detections
                    )
                    positive_annotations_files[light_type].write(
                        "{} {} {}\n".format(
                            image_relative, len(detections), detections_string
                        )
                    )

                # Append to the negative annotations file
                for light_type in dataset.classes:
                    if light_type not in light_detections.keys():
                        negative_annotations_files[light_type].write(
                            f"{new_image_path}\n"
                        )

        # Close the positive and negative annotation files
        for file in positive_annotations_files.values():
            file.close()
        for file in negative_annotations_files.values():
            file.close()

        # Generate the light type to absolute annotations path mapping
        positive_annotations = {
            light_type: os.path.join(annotations_folder, file.name)
            for light_type, file in positive_annotations_files.items()
        }
        negative_annotations = {
            light_type: os.path.join(annotations_folder, file.name)
            for light_type, file in negative_annotations_files.items()
        }

        return HaarData(positive_annotations, negative_annotations)
