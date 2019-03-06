from typing import Dict, List

import os
import copy
import random

import cv2  # type: ignore

from lns.common import config


def generate_resources_root() -> None:
    """Generate the folder structure for the common resources."""
    if not os.path.isdir(config.RESOURCES_ROOT):
        os.makedirs(config.RESOURCES_ROOT)


class Data:
    """Stores information about available data."""

    # Absolute path to the data folder
    _DEFAULT_DATA_ROOT: str = os.path.join(config.RESOURCES_ROOT, "data")

    # Current data root and stored datsets
    _data_root: str = _DEFAULT_DATA_ROOT
    _datasets: Dict[str, str] = {}

    @classmethod
    def set_data_root(cls, new_root: str = _DEFAULT_DATA_ROOT, force_create: bool = False) -> None:
        """Set the data root folder.

        Creates the data root if it does not exist if <force_create> is set to
        be `True`, otherwise raises `ValueError` if there is any discrepency in
        the new data root.
        """
        # Create the data folder if the flag is set
        if force_create and not os.path.exists(new_root):
            os.makedirs(new_root)

        # Ensure the new data root exists
        new_data_root: str = os.path.abspath(new_root)
        if not os.path.isdir(new_data_root):
            raise ValueError(f"Proposed data root `{new_data_root}` is not a valid directory.")

        # Populate the new datasets available in this data root
        new_datasets: Dict[str, str] = {
            folder_name: os.path.join(new_data_root, folder_name)
            for folder_name in os.listdir(new_data_root)
            if os.path.isdir(os.path.join(new_data_root, folder_name)) and folder_name in config.POSSIBLE_DATASETS
        }

        # Assign the proposed data root and datasets
        cls._data_root = new_data_root
        cls._datasets = new_datasets

    @classmethod
    def get_data_root(cls) -> str:
        """Get the path to the root of the data folder."""
        return cls._data_root

    @classmethod
    def get_dataset_path(cls, dataset_name: str) -> str:
        """Get the path to the dataset with the given <dataset_name>.

        Raises `KeyError` if no such dataset exists.
        """
        try:
            return cls._datasets[dataset_name]
        except KeyError as e:
            raise e

    @classmethod
    def has_dataset(cls, dataset_name: str) -> bool:
        """Get whether or not a dataset with the given name is available."""
        try:
            cls.get_dataset_path(dataset_name)
        except KeyError:
            return False
        return True


class Dataset:
    """Read-only container structure for data generated by preprocessing."""

    _name: str

    Images = Dict[str, List[str]]
    Classes = List[str]
    Labels = List[Dict[str, int]]
    Annotations = Dict[str, Labels]

    __images: Images
    __classes: Classes
    __annotations: Annotations

    def __init__(self, name: str, images: 'Dataset.Images', classes: 'Dataset.Classes',
                 annotations: 'Dataset.Annotations', shuffle: bool = True) -> None:
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
        self.__images = copy.deepcopy(images)
        self.__classes = copy.deepcopy(classes)
        self.__annotations = copy.deepcopy(annotations)

        if shuffle:
            self.shuffle_images(seed=config.SEED)

    @property
    def name(self) -> str:
        """Get the name of this dataset."""
        return self._name

    @property
    def images(self) -> 'Dataset.Images':
        """Get a list of paths to all images available in the dataset."""
        return copy.deepcopy(self.__images)

    @property
    def classes(self) -> 'Dataset.Classes':
        """Get a mapping of ID to name for all classes in the dataset."""
        return copy.deepcopy(self.__classes)

    @property
    def annotations(self) -> 'Dataset.Annotations':
        """Get all training image annotations.

        Image annotations are structured as a mapping of absolute image path
        (as given in `self.images`) to a list of detections. Each detection
        consists of a mapping from detection key to its respective information.

        Available detection keys are
        `class`, `x_min`, `y_min`, `x_max`, `y_max`.
        """
        return copy.deepcopy(self.__annotations)

    def image_split(self, *proportions: float) -> List['Dataset.Images']:
        """Get subsets of images split based on the given proportions.

        Returns a dictionary of dataset name to list of lists of image paths
        each with given proportions of images from the total dataset
        that are non-overlapping.
        """
        if sum(proportions) > 1:
            raise ValueError(f"Got total proportion {sum(proportions)} > 1.")
        if any(proportion < 0 for proportion in proportions):
            raise ValueError("No proportion can be negative.")

        images = self.images
        num_images = {
            dataset_name: len(image_paths)
            for dataset_name, image_paths in images.items()
        }
        count_split = {
            dataset_name: [int(num_images[dataset_name] * proportion) for proportion in proportions]
            for dataset_name in images.keys()
        }
        segmentation = {
            dataset_name: [0] + [sum(count_split[dataset_name][:i + 1]) for i, _ in enumerate(proportions)]
            for dataset_name in images.keys()
        }

        return [{
            dataset_name: image_paths[segmentation[dataset_name][i]:segmentation[dataset_name][i + 1]]
            for dataset_name, image_paths in images.items()
        } for i, _ in enumerate(proportions)]

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

    def shuffle_images(self, seed: int) -> None:
        """Shuffle the images in this `Dataset` in place.

        Will set the random seed if given `seed`.
        """
        random.seed(seed)
        for dataset_name in self.__images.keys():
            random.shuffle(self.__images[dataset_name])

    def scale(self, scale: float) -> 'Dataset':
        """Generate a new dataset with all images scaled by `scale`."""
        images: Dataset.Images = self.images
        classes: Dataset.Classes = self.classes
        annotations: Dataset.Annotations = self.annotations

        new_images: Dataset.Images = {}
        new_name = f"{self.name}_{scale}"
        new_path = os.path.join(Data.get_data_root(), new_name)
        if not os.path.isdir(new_path):
            os.makedirs(new_path)

        for image_path, annotation in self.annotations.items():
            for i, label in enumerate(annotation):
                label['x_min'] = int(label['x_min'] * scale)
                label['y_min'] = int(label['y_min'] * scale)
                label['x_max'] = int(label['x_max'] * scale)
                label['y_max'] = int(label['y_max'] * scale)

            new_image_path = os.path.join(new_path, os.path.basename(image_path))
            annotations[new_image_path] = annotations[image_path]
            del annotations[image_path]

        for name, image_paths in images.items():
            new_images[name] = []
            for i, image_path in enumerate(image_paths):
                new_image_path = os.path.join(new_path, os.path.basename(image_path))
                if os.path.isfile(new_image_path):
                    continue

                new_image = cv2.imread(image_path).resize((0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                new_image.save(new_image_path)
                new_images[name].append(new_image_path)

        return Dataset(new_name, new_images, classes, annotations)

    def __add__(self, other: 'Dataset') -> 'Dataset':
        """Magic method for adding two preprocessing data objects."""
        return Dataset(f"{self.name}-{other.name}",
                       {**self.images, **other.images},
                       self.classes + other.classes,
                       {**self.annotations, **other.annotations})

    def __len__(self) -> int:
        """Magic method to get the length of this `Dataset`.

        We define the length of a dataset to the the total number of images.
        """
        return sum(len(image_paths) for image_paths in self.__images.values())


generate_resources_root()
Data.set_data_root(force_create=True)
