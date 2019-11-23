"""Data processing for SqueezeDet training.

Manages all data processing for the generation of data ready to be trained
on with SqueezeDet Keras fitting functions.
"""
import os
from typing import List

from lns.common.dataset import Dataset
from lns.common.process import ProcessedData, Processor


class SqueezedetData(ProcessedData):
    """Data container for all SqueezeDet processed data.

    Store paths to all files needed by backend SqueezeDet training.
    """

    __images: str
    __labels: str

    def __init__(self, images: str, labels: str) -> None:
        """Initialize the data structure."""
        self.__images = images
        self.__labels = labels

    def get_images(self) -> str:
        """Get a path to the text file containing paths to all images."""
        return self.__images

    def get_labels(self) -> str:
        """Get a path to the text file containing paths to all labels."""
        return self.__labels


class SqueezedetProcessor(Processor[SqueezedetData]):
    """Squeezedet processor responsible for data processing to Squeezedet-valid formats."""

    @classmethod
    def method(cls) -> str:
        """Get the training method this processor is for."""
        return "squeezedet"

    @classmethod
    def _process(cls, dataset: Dataset) -> SqueezedetData:  # pylint: disable=too-many-locals
        # Register all folders
        processed_data_folder = os.path.join(cls.get_processed_data_path(), dataset.name)
        labels_folder = os.path.join(processed_data_folder, "labels")

        # Create labels folder if doesn't exist already
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)

        # Generate the labels files corresponding to the images
        images: List[str] = []
        labels: List[str] = []
        for image, annotations in dataset.annotations.items():
            label_strings: List[str] = []
            for annotation in annotations:
                if (annotation.bounds.left < 0 or annotation.bounds.top < 0
                   or annotation.bounds.left > annotation.bounds.right
                   or annotation.bounds.top > annotation.bounds.bottom):
                    continue

                # NOTE: see https://github.com/NVIDIA/DIGITS/issues/992
                # for more information about the format
                class_name = "".join(dataset.classes[annotation.class_index].lower().split())
                label_strings.append(" ".join((
                    str(class_name),                       # class string
                    "0",                                   # truncation number
                    "0",                                   # occlusion number
                    "0",                                   # observation angle
                    str(annotation.bounds.left),           # left
                    str(annotation.bounds.top),            # top
                    str(annotation.bounds.right),          # right
                    str(annotation.bounds.bottom),         # bottom
                    "0",                                   # height (3d)
                    "0",                                   # width  (3d)
                    "0",                                   # length (3d)
                    "0",                                   # x loc  (3d)
                    "0",                                   # y loc  (3d)
                    "0",                                   # z loc  (3d)
                    "0",                                   # y rot  (3d)
                    "0"                                    # score
                )))

            # Do not write empty files
            if not label_strings:
                continue

            images.append(image)

            # Create the file and put the strings in it
            # XXX: might cause issues if different files with same basename exist
            label = "".join(os.path.basename(image).split(".")[:-1]) + ".txt"
            label_path = os.path.join(labels_folder, label)
            with open(label_path, "w") as label_file:
                label_file.write("\n".join(label_strings))
            labels.append(label_path)

        # Sort the images and labels lexicographically
        images = sorted(images, key=lambda image: image.split("/")[-1])
        labels = sorted(labels, key=lambda image: image.split("/")[-1])

        # Create images and labels files
        labels_path = os.path.join(processed_data_folder, "labels.txt")
        with open(labels_path, "w") as labels_file:
            labels_file.write("\n".join(labels))
        images_path = os.path.join(processed_data_folder, "images.txt")
        with open(images_path, "w") as images_file:
            images_file.write("\n".join(images))

        return SqueezedetData(images_path, labels_path)


SqueezedetProcessor.init_cached_processed_data()
