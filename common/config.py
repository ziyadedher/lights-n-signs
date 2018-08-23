"""General configuration file.

Use this file instead of hard-coding any directories or for any other general
configuration of the package.
"""
from typing import Dict, List

import os


class Data:
    """Stores information about available data."""

    # Absolute path to the data folder
    _DEFAULT_DATA_ROOT: str = os.path.abspath(
        os.path.join(__file__, os.pardir, "data")
    )

    # Possible datasets available
    _POSSIBLE_DATASET_FOLDERS: List[str] = ["LISA"]

    # Current data root and stored datsets
    _data_root: str = _DEFAULT_DATA_ROOT
    _datasets: Dict[str, str] = {}

    @classmethod
    def set_data_root(cls, new_root: str = _DEFAULT_DATA_ROOT) -> None:
        """Set the data root folder.

        Raises `ValueError` if there is any discrepency in the new data root.
        """
        # Ensure the new data root exists
        new_data_root: str = os.path.abspath(new_root)
        if not os.path.isdir(new_data_root):
            raise ValueError(
                f"Proposed data root `{new_data_root}` " +
                "is not a valid directory."
            )

        # Populate the new datasets available in this data root
        new_datasets: Dict[str, str] = {
            folder_name: os.path.join(new_data_root, folder_name)
            for folder_name in os.listdir(new_data_root)
            if os.path.isdir(os.path.join(new_data_root, folder_name))
            and folder_name in cls._POSSIBLE_DATASET_FOLDERS
        }

        # Assign the proposed data root and datasets
        cls._data_root = new_data_root
        cls._datasets = new_datasets

    @classmethod
    def get_dataset_path(cls, dataset_name: str) -> str:
        """Get the path to the dataset with the given <dataset_name>.

        Raises `NoSuchDatasetException` if no such dataset exists.
        """
        try:
            return cls._datasets[dataset_name]
        except KeyError:
            raise NoSuchDatasetException(dataset_name)


class NoSuchDatasetException(Exception):
    """Raised when a dataset not in the data root is referenced."""

    pass


Data.set_data_root()
