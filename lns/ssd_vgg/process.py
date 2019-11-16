"""Data processing for SsdVgg training.

Manages all data processing for the generation of data ready to be trained
on with SSD VGG.
"""
from typing import List

import os
import shutil

from PIL import Image # type: ignore
from lns.common import config
from lns.common.process import ProcessedData, Processor
from lns.common.preprocess.preprocessing import Dataset
from xml.dom import minidom  # type: ignore
import xml.etree.cElementTree as ET


class SsdVggData(ProcessedData):
    """Data container for all SSD VGG processed data."""

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

    #def generate_data(self, _config: easydict.EasyDict) -> Generator[
    #    Any, None, None
    #]:
    #    """Generate data in the format required for training SsdVgg.

    #    Requires the configuration dictionary that the model to be trained is
    #    based off of.
    #    """
    #    yield from dataGenerator.generator_from_data_path(
    #        self.images, self.labels, config=_config
    #    )


class SsdVggProcessor(Processor[SsdVggData]):
    """SSD VGG processor responsible for data processing to VGG-valid formats."""

    BASE_DATA_FOLDER = os.path.join(config.RESOURCES_ROOT, "ssdvgg/data")

    @classmethod
    def process(cls, dataset: Dataset, force: bool = False) -> SsdVggData:
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
            label_strings: List[str] = []
            im_width, im_height = 0, 0
            with Image.open(image) as img:
                im_width, im_height = img.size
            top = ET.Element('annotation')
            filename = ET.SubElement(top, 'filename')
            filename.text = str(image)
            size = ET.SubElement(top, 'size')
            width = ET.SubElement(size, 'width')
            width.text = str(im_width)
            height = ET.SubElement(size, 'height')
            height.text = str(im_height)

            for annotation in annotations:
                if (annotation["x_min"] < 0 or annotation["y_min"] < 0
                    or annotation["x_min"] > annotation["x_max"]
                    or annotation["y_min"] > annotation["y_max"]):
                    continue

                # NOTE: see https://github.com/NVIDIA/DIGITS/issues/992
                # for more information about the format
                class_name = "".join(dataset.classes[annotation["class"]].lower().split())
                obj = ET.SubElement(top, 'object')
                name = ET.SubElement(obj, 'name')
                name.text = str(class_name)
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                ymin = ET.SubElement(bndbox, 'ymin')
                xmax = ET.SubElement(bndbox, 'xmax')
                ymax = ET.SubElement(bndbox, 'ymax')
                xmin.text = str(annotation["x_min"])
                ymin.text = str(annotation["y_min"])
                xmax.text = str(annotation["x_max"])
                ymax.text = str(annotation["y_max"])
            rough_string = ET.tostring(top, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            label_strings.append(reparsed.toprettyxml(indent='\t'))
            # Do not write empty files
            if len(label_strings) == 0:
                continue

            images.append(image)

            # Create the file and put the strings in it
            label = "".join(os.path.basename(image).split(".")[:-1]) + ".txt"
            label_path = os.path.join(labels_folder, label)
            with open(label_path, "w") as label_file:
                label_file.write("\n".join(label_strings))
            labels.append(label_path)

        # Sort the images and labels
        images = sorted(images, key=lambda image: image.split("/")[-1])
        labels = sorted(labels, key=lambda image: image.split("/")[-1])

        # Create images and labels files
        labels_path = os.path.join(data_folder, "labels.txt")
        with open(labels_path, "w") as labels_file:
            labels_file.write("\n".join(labels))
        images_path = os.path.join(data_folder, "images.txt")
        with open(images_path, "w") as images_file:
            images_file.write("\n".join(images))

        return SsdVggData(images, labels)
