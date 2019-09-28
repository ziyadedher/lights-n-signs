"""Data processing for Haar cascade training.

Manages all data processing for the generation of data ready to be trained
on with OpenCV Haar training scripts.
"""
from typing import ClassVar, List

import os

import cv2             # type: ignore
from tqdm import tqdm  # type: ignore

from lns.common.structs import Object2D
from lns.common.dataset import Dataset
from lns.common.process import ProcessedData, Processor


class HaarData(ProcessedData):
    """Data container for all Haar processed data.

    Contains positive annotations for each type of class and negative
    annotations for each type of class as well from the dataset.
    """

    __positive_annotations: List[str]
    __negative_annotations: List[str]

    def __init__(self, positive_annotations: List[str], negative_annotations: List[str]) -> None:
        """Initialize the data structure."""
        self.__positive_annotations = positive_annotations
        self.__negative_annotations = negative_annotations

    def get_positive_annotation(self, class_index: int) -> str:
        """Get the path to a positive annotation file for the given class index.

        Raises `KeyError` if no such class index is available.
        """
        try:
            return self.__positive_annotations[class_index]
        except IndexError as err:
            raise err

    def get_negative_annotation(self, class_index: int) -> str:
        """Get the path to a negative annotation file for the given class index.

        Raises `KeyError` if no such class index is available.
        """
        try:
            return self.__negative_annotations[class_index]
        except IndexError as err:
            raise err


class HaarProcessor(Processor[HaarData]):
    """Haar processor responsible for data processing to Haar-valid formats."""

    METHOD: ClassVar[str] = "haar"

    @classmethod
    def method(cls) -> str:
        """Get the training method this processor is for."""
        return cls.METHOD

    @classmethod  # noqa: R701
    def _process(cls, dataset: Dataset) -> HaarData:  # noqa: R914
        # Register all folders
        processed_data_folder = os.path.join(cls.get_processed_data_path(), dataset.name)
        annotations_folder = os.path.join(processed_data_folder, "annotations")
        images_folder = os.path.join(processed_data_folder, "images")
        os.makedirs(annotations_folder)
        os.makedirs(images_folder)

        # Open the positive and negative annotation files
        pos_files = [open(os.path.join(annotations_folder, name + "_positive"), "w") for name in dataset.classes]
        neg_files = [open(os.path.join(annotations_folder, name + "_negative"), "w") for name in dataset.classes]

        # Set up for reading annotations
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        # Read all annotations
        with tqdm(desc="Processing", total=len(dataset.annotations.keys()), miniters=1) as tqdm_bar:
            for i, (image_path, labels) in enumerate(dataset.annotations.items()):
                # Update the progress bar
                tqdm_bar.update()

                # Create gray images
                new_image_path = os.path.join(images_folder, f"{i}.png")
                cv2.imwrite(new_image_path, clahe.apply(cv2.imread(image_path, 0)))

                image_relative = os.path.relpath(os.path.join(images_folder, f"{i}.png"), start=annotations_folder)

                # Store the annotations in a way easier to represent for Haar
                class_annotations = cls._reformat_labels(labels, dataset)

                for j, annotations in enumerate(class_annotations):
                    detections_string = " ".join(
                        " ".join(str(item) for item in annotation) for annotation in annotations)
                    pos_files[j].write(f"{image_relative} {len(annotations)} {detections_string}\n")
                for j, _ in enumerate(dataset.classes):
                    if i not in class_annotations:
                        neg_files[j].write(f"{new_image_path}\n")

        # Close the positive and negative annotation files
        for file in pos_files:
            file.close()
        for file in neg_files:
            file.close()

        # Generate the light type to absolute annotations path mapping
        positive_annotations = [os.path.join(annotations_folder, file.name) for file in pos_files]
        negative_annotations = [os.path.join(annotations_folder, file.name) for file in neg_files]

        return HaarData(positive_annotations, negative_annotations)

    @classmethod
    def _reformat_labels(cls, labels: List[Object2D], dataset: Dataset) -> List[List[List[float]]]:
        annotations: List[List[List[float]]] = [[] for _ in dataset.classes]
        for label in labels:
            class_index = label.class_index
            x_min = label.bounds.left
            y_min = label.bounds.top
            width = label.bounds.width
            height = label.bounds.height
            annotations[class_index].append([x_min, y_min, width, height])
        return annotations


HaarProcessor.init_cached_processed_data()
