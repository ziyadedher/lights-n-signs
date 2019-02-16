"""General configuration file.

Use this file instead of hard-coding any directories or for any other general
configuration of the package.
"""
from typing import Dict, List

import os


RESOURCES_ROOT = os.path.abspath(
    os.path.join(os.path.expanduser("~"), ".lns-training/resources")
)

RAND_SEED = 6
MIN_SIZE = 40

def generate_resources_root() -> None:
    """Generate the folder structure for the common resources."""
    if not os.path.isdir(RESOURCES_ROOT):
        os.makedirs(RESOURCES_ROOT)


class Data:
    """Stores information about available data."""

    # Absolute path to the data folder
    _DEFAULT_DATA_ROOT: str = os.path.join(RESOURCES_ROOT, "data")

    # Possible datasets available
    _POSSIBLE_DATASET_FOLDERS: List[str] = [
        "LISA", "Bosch", "Custom", "Custom_testing", "sim", "mturk", "cities", "LISA_signs", "lights"
    ]

    # Current data root and stored datsets
    _data_root: str = _DEFAULT_DATA_ROOT
    _datasets: Dict[str, str] = {}

    @classmethod
    def set_data_root(cls,
                      new_root: str = _DEFAULT_DATA_ROOT,
                      force_create: bool = False) -> None:
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


generate_resources_root()
Data.set_data_root(force_create=True)
