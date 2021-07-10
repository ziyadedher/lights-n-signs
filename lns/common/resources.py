"""Manages representation of disk resources for the pipeline.

Resources include plain data, cached datasets, and other large objects.
"""

from typing import Dict

import os

from lns.common import config


class Resources:
    """Stores information about available resources."""

    # Current data root and stored datsets
    _data_root: str = config.RESOURCES_ROOT
    _datasets: Dict[str, str] = {}

    @classmethod
    def set_root(cls, new_root: str = config.RESOURCES_ROOT, force_create: bool = False) -> None:
        """Set the resources root folder.

        Creates the resources root if it does not exist if <force_create> is set to
        be `True`, otherwise raises `ValueError` if there is any discrepency in
        the new data root.
        """
        new_root_path = os.path.abspath(new_root)

        # Create the data folder if the flag is set
        if force_create and not os.path.exists(new_root_path):
            config.generate_resources_filetree(new_root_path)

        # Ensure the new data root exists
        if not os.path.isdir(new_root_path):
            raise ValueError(f"Proposed data root `{new_root_path}` is not a valid directory.")

        # Populate the new datasets available in the data root
        new_data_root = os.path.join(new_root_path, config.DATA_FOLDER_NAME)
        new_datasets: Dict[str, str] = {
            folder_name: os.path.join(new_data_root, folder_name)
            for folder_name in os.listdir(new_data_root)
            if os.path.isdir(os.path.join(new_data_root, folder_name)) and folder_name in config.POSSIBLE_DATASETS
        }

        # Assign the proposed data root and datasets
        cls._data_root = new_root_path
        cls._datasets = new_datasets

    @classmethod
    def get_root(cls) -> str:
        """Get the path to the root of the resources folder."""
        return cls._data_root

    @classmethod
    def get_dataset_path(cls, dataset_name: str) -> str:
        """Get the path to the dataset with the given <dataset_name>.

        Raises `KeyError` if no such dataset exists.
        """
        try:
            return cls._datasets[dataset_name]
        except KeyError as err:
            raise err

    @classmethod
    def has_dataset(cls, dataset_name: str) -> bool:
        """Get whether or not a dataset with the given name is available."""
        try:
            cls.get_dataset_path(dataset_name)
        except KeyError:
            return False
        return True


Resources.set_root()
