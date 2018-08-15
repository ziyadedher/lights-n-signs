"""General configuration file.

Use this file instead of hard-coding any directories or for any other general
configuration of the package.
"""
from types import ModuleType
from typing import Optional, Union

import os


# Absolute path to the data folder
__DEFAULT_DATA_ROOT = os.path.join(os.path.abspath(__file__), "data")

# Names of the dataset folders inside of the main data root
__DATASET_FOLDER_NAMES = ["LISA"]


class NoSuchDatasetException(Exception):
    """Raised when a dataset not in the data root is referenced."""

    pass


def get_opencv_bindings() -> ModuleType:
    """Get OpenCV bindings module if it exists.

    Raises `ImportError` if the bindings could not be important.
    """
    try:
        import cv2
        return cv2
    except ImportError as e:
        print("Could not import OpenCV Python bindings `cv2`, \
              please ensure that OpenCV for Python is correctly installed.")
        raise e


def set_data_root(new_root: Union[str, os.PathLike]) -> None:
    """Set the data root folder.

    Raises `ValueError` if there is any discrepency in the new data root.
    """
    # Generate the proposed new data root and dataset paths
    new_data_root = os.path.abspath(new_root)
    new_datasets = {
        folder_name: os.path.join(new_data_root, folder_name)
        for folder_name in __DATASET_FOLDER_NAMES.keys()
    }

    # Ensure the new data root exists
    if not os.path.isdir(new_data_root):
        raise ValueError(
            f"Proposed data root `{new_data_root}` is not a valid directory."
        )

    # Ensure all datasets are present in the new data root
    for folder_name in __DATASET_FOLDER_NAMES:
        path = os.path.join(new_data_root, folder_name)
        if not os.path.isdir(path):
            raise ValueError(
                f"Dataset folder `{folder_name}` \
                is not in the proposed data root `{new_data_root}`."
            )

    # Assign the proposed data root and datasets
    global __DATA_ROOT
    global __DATASETS
    __DATA_ROOT = new_data_root
    __DATASETS = new_datasets


def get_dataset_path(dataset_name: str) -> Optional[str]:
    """Get the path to the dataset with the given <dataset_name>.

    Raises `NoSuchDatasetException` if no such dataset exists.
    """
    try:
        return __DATASETS[dataset_name]
    except KeyError:
        raise NoSuchDatasetException(dataset_name)


__DATA_ROOT = ""
__DATASETS = {}
set_data_root(__DEFAULT_DATA_ROOT)
