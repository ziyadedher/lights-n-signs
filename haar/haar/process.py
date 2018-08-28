"""Data processing for Haar cascade training.

Manages all data processing for the generation of data ready to be trained
on with OpenCV Haar training scripts.
"""
from typing import List, Dict

import os
import shutil

import cv2             # type: ignore
from tqdm import tqdm  # type: ignore

from common import config
from common.preprocess.preprocess import Preprocessor, NoPreprocessorException


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

    _processing_data: Dict[str, HaarData] = {}

    @classmethod
    def process(cls, dataset_name: str,
                force: bool = False) -> HaarData:
        """Process all required data from the dataset with the given name.

        Setting <force> to `True` will force a preprocessing even if the
        preprocessed data already exists in memory.

        Raises `NoSuchDatasetException` if such a dataset does not exist.
        Raises `NoPreprocessorException` if a preprocessor for the dataset does
        not exist.
        """
        # TODO: structure this function better
        # TODO: read any data that exists on file as well
        # Uses memoization to speed up processing acquisition
        if not force and dataset_name in cls._processing_data:
            return cls._processing_data[dataset_name]

        try:
            preprocessed_data = Preprocessor.preprocess(dataset_name)
        except config.NoSuchDatasetException as e:
            raise e
        except NoPreprocessorException as e:
            raise e

        # Remove and generate required folders
        base_data_folder = os.path.join(config.RESOURCES_ROOT, "haar/data")
        data_folder = os.path.join(base_data_folder, dataset_name)
        annotations_folder = os.path.join(data_folder, "annotations")
        images_folder = os.path.join(data_folder, "images")

        # Create required folders if they do not exist
        if not os.path.exists(base_data_folder):
            os.mkdir(base_data_folder)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)

        # Remove annotations folder to be regenerated
        if os.path.exists(annotations_folder):
            shutil.rmtree(annotations_folder)
        os.mkdir(annotations_folder)

        # Open the positive and negative annotation files
        positive_annotations_files = {
            light_type: open(os.path.join(
                annotations_folder, light_type + "_positive"
            ), "w") for light_type in preprocessed_data.classes
        }
        negative_annotations_files = {
            light_type: open(os.path.join(
                annotations_folder, light_type + "_negative"
            ), "w") for light_type in preprocessed_data.classes
        }

        # Set up for reading annotations
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enumeration = enumerate(preprocessed_data.annotations.items())

        # Read all annotations
        with tqdm(desc="Preprocessing",
                  total=len(preprocessed_data.annotations.keys()),
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
                    class_name = preprocessed_data.classes[label["class"]]
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
                for light_type in preprocessed_data.classes:
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

        # Memoize and return the processed data
        processed_data = HaarData(positive_annotations, negative_annotations)
        cls._processing_data[dataset_name] = processed_data
        return processed_data
