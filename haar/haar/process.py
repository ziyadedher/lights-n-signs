"""Data processing for Haar cascade training.

Manages all data processing for the generation of data ready to be trained
on with OpenCV Haar training scripts.
"""
from typing import List, Dict

import os
import shutil

from common.config import NoSuchDatasetException
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
        # Uses memoization to speed up processing acquisition
        if not force and dataset_name in cls._processing_data:
            return cls._processing_data[dataset_name]

        try:
            preprocessed_data = Preprocessor.preprocess(dataset_name)
        except NoSuchDatasetException as e:
            raise e
        except NoPreprocessorException as e:
            raise e

        # Remove the annotations folder if it exists and create it
        base_annotations_folder_path = os.path.abspath(os.path.join(
            __file__, os.pardir, "annotations"
        ))
        annotations_folder_path = os.path.join(
            base_annotations_folder_path, dataset_name
        )
        if not os.path.exists(base_annotations_folder_path):
            os.mkdir(base_annotations_folder_path)
        if os.path.exists(annotations_folder_path):
            shutil.rmtree(annotations_folder_path)
        os.mkdir(annotations_folder_path)

        # Open the positive and negative annotation files
        positive_annotations_files = {
            light_type: open(os.path.join(
                annotations_folder_path, light_type + "_positive"
            ), "w") for light_type in preprocessed_data.classes
        }
        negative_annotations_files = {
            light_type: open(os.path.join(
                annotations_folder_path, light_type + "_negative"
            ), "w") for light_type in preprocessed_data.classes
        }

        # Read all annotations
        for image_path, annotations in preprocessed_data.annotations.items():
            # Get the relative image path for storing in the file
            image_relative = os.path.relpath(
                image_path, start=os.path.join(annotations_folder_path)
            )

            # Store the annotations in a way easier to represent for Haar
            light_detections: Dict[str, List[List[int]]] = {}

            # Go through each detection and populate the above dictionary
            for annotation in annotations:
                class_name = preprocessed_data.classes[annotation["class"]]
                x_min = annotation["x_min"]
                y_min = annotation["y_min"]
                width = annotation["x_max"] - x_min
                height = annotation["y_max"] - y_min

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
                    f"{image_relative} {len(detections)} {detections_string}\n"
                )

            # Append to the negative annotations file
            for light_type in preprocessed_data.classes:
                if light_type not in light_detections.keys():
                    negative_annotations_files[light_type].write(
                        f"{image_relative}\n"
                    )

        # Close the positive and negative annotation files
        for file in positive_annotations_files.values():
            file.close()
        for file in negative_annotations_files.values():
            file.close()

        # Generate the light type to absolute annotations path mapping
        positive_annotations = {
            light_type: os.path.join(annotations_folder_path, file.name)
            for light_type, file in positive_annotations_files.items()
        }
        negative_annotations = {
            light_type: os.path.join(annotations_folder_path, file.name)
            for light_type, file in negative_annotations_files.items()
        }

        # Memoize and return the processed data
        processed_data = HaarData(positive_annotations, negative_annotations)
        cls._processing_data[dataset_name] = processed_data
        return processed_data
