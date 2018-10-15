"""Data processing for SqueezeDet training.

Manages all data processing for the generation of data ready to be trained
on with SqueezeDet Keras fitting functions.
"""
from typing import Generator, Any, List

import os
import shutil

import easydict                                   # type: ignore
from squeezedet_keras.model import dataGenerator  # type: ignore

from lns_common import config
from lns_common.process import ProcessedData, Processor
from lns_common.preprocess.preprocessing import Dataset


class SqueezeDetData(ProcessedData):
    """Data container for all SqueezeDet processed data."""

    __images: List[str]
    __labels: List[str]

    def __init__(self, images: List[str], labels: List[str]) -> None:
        """Initialize the data structure."""
        self.__images = images
        self.__labels = labels

    @property
    def images(self) -> List[str]:
        """Get a list of paths to all images."""
        return self.__images

    @property
    def labels(self) -> List[str]:
        """Get a list of paths to all ground truths."""
        return self.__labels

    def generate_data(self, _config: easydict.EasyDict) -> Generator[
        Any, None, None
    ]:
        """Generate data in the format required for training SqueezeDet.

        Requires the configuration dictionary that the model to be trained is
        based off of.
        """
        yield from dataGenerator.generator_from_data_path(
            self.images, self.labels, config=_config
        )


class SqueezeDetProcessor(Processor[SqueezeDetData]):
    """Haar processor responsible for data processing to Haar-valid formats."""

    BASE_DATA_FOLDER = os.path.join(config.RESOURCES_ROOT, "squeezedet/data")

    @classmethod
    def process(cls, dataset: Dataset, force: bool = False) -> SqueezeDetData:
        """Process all required data from the dataset with the given name.

        Setting <force> to `True` will force a processing even if the images
        already exist on file.

        Raises `NoPreprocessorException` if a preprocessor for the dataset does
        not exist.
        """
        # Register all folders
        data_folder = os.path.join(cls.BASE_DATA_FOLDER, dataset.name)
        labels_folder = os.path.join(data_folder, "labels")

        # Create base data folder if it does not exist
        if not os.path.exists(cls.BASE_DATA_FOLDER):
            os.makedirs(cls.BASE_DATA_FOLDER)

        # Remove labels folder if required to be regenerated
        if os.path.exists(labels_folder) and force:
            shutil.rmtree(labels_folder)
        elif not os.path.exists(labels_folder):
            os.makedirs(labels_folder)

        # Generate the labels files corresponding to the images
        images: List[str] = []
        labels: List[str] = []
        for image, annotations in dataset.annotations.items():
            images.append(image)
            label_strings: List[str] = []
            for annotation in annotations:
                # NOTE: see https://github.com/NVIDIA/DIGITS/issues/992
                # for more information about the format
                label_strings.append(" ".join((
                    dataset.classes[annotation["class"]],  # class string
                    "0",                                   # truncation number
                    "0",                                   # occlusion number
                    "0",                                   # observation angle
                    str(annotation["x_min"]),              # left
                    str(annotation["y_min"]),              # top
                    str(annotation["x_max"]),              # right
                    str(annotation["y_max"]),              # bottom
                    "0",                                   # height (3d)
                    "0",                                   # width  (3d)
                    "0",                                   # length (3d)
                    "0",                                   # x loc  (3d)
                    "0",                                   # y loc  (3d)
                    "0",                                   # z loc  (3d)
                    "0",                                   # y rot  (3d)
                    "0"                                    # score
                )))

            # Create the file and put the strings in it
            label = "".join(os.path.basename(image).split(".")[:-1]) + ".txt"
            label_path = os.path.join(labels_folder, label)
            with open(label_path, "w") as label_file:
                label_file.write("\n".join(label_strings))
            labels.append(label_path)

        return SqueezeDetData(images, labels)
