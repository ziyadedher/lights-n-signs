
"""Data Processing for SVM training.

Manages all data processing to generate data that is ready to be trained on by the
Open CV SVM library
"""
from typing import ClassVar

import numpy as np
import cv2
import os

from lns.common.structs import crop
from lns.common.dataset import Dataset
from lns.common.process import ProcessedData, Processor


class SVMData(ProcessedData):
    """Data container for the SVM processed data.

    Contains Images and Labels
    """

    __images: str
    __labels: str

    def __init__(self, images: str, labels: str) -> None:
        """Initialize the structure."""
        self.__images = images
        self.__labels = labels

    @property
    def get_images(self) -> str:
        """2D array to store images, shape = (number of images, size of image)."""
        return self.__images

    @property
    def get_labels(self) -> str:
        """Get array of labels."""
        return self.__labels


class SVMProcessor(Processor):
    """Processor for processing to SVM training format."""

    METHOD: ClassVar[str] = "svm"

    @classmethod
    def method(cls) -> str:
        """Return the training method."""
        return cls.METHOD

    @classmethod
    def _process(cls, dataset: Dataset) -> SVMData:
        """Load the images and save them as numpy arrays."""
        processed_data_folder = os.path.join(cls.get_processed_data_path(), dataset.name)
        images_file = os.path.join(processed_data_folder, "images.npy")
        labels_file = os.path.join(processed_data_folder, "labels.npy")

        svm_images = np.array([])
        svm_labels = np.array([])

        if os.path.exists(images_file) and os.path.exists(labels_file):
            return SVMData(images_file, labels_file)

        for image_file in dataset.images:
            labels = dataset.annotations[image_file]
            im = cv2.imread(image_file)

            for label in labels:
                box = crop(im, label.bounds)
                box = cv2.resize(box, (32, 32))
                box = box.flatten()  # Flatten so that it can be used for SVM training
                svm_images = np.append(svm_images, [box])
                svm_labels = np.append(svm_labels, label.class_index)

        svm_images = svm_images.reshape(len(svm_labels), 3072)

        np.save(images_file, svm_images)
        np.save(labels_file, svm_labels)

        return SVMData(images_file, labels_file)
