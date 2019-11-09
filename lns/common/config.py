"""General configuration file.

Use this file instead of hard-coding any directories or for any other general
configuration of the package.
"""
from typing import List

import os


RESOURCES_ROOT: str = os.path.abspath(os.path.join(os.path.expanduser("~"), ".lns-training/resources"))

DATA_FOLDER_NAME = "data"
PREPROCESSED_DATA_FOLDER_NAME = "datasets"
TRAINERS_FOLDER_NAME = "trainers"
PROCESSED_DATA_FOLDER_NAME = "processed"
WEIGHTS_FOLDER_NAME = "weights"
DATA_DISTANCES_FOLDER_NAME = "dataset_distances"
SUBROOT_FOLDERS = [
    DATA_FOLDER_NAME,
    PREPROCESSED_DATA_FOLDER_NAME,
    TRAINERS_FOLDER_NAME,
    PROCESSED_DATA_FOLDER_NAME,
    WEIGHTS_FOLDER_NAME,
    DATA_DISTANCES_FOLDER_NAMEA
]

PKL_EXTENSION = ".pkl"

POSSIBLE_DATASETS: List[str] = [
    "mocked",
    "LISA",
    "Bosch",
    "ScaleLights",
    "ScaleSigns",
    "ScaleObjects"
]


def generate_resources_filetree(root: str = RESOURCES_ROOT) -> None:
    """Generate the folder structure for all the common resources."""
    if not os.path.isdir(root):
        os.makedirs(root)
    for subroot in SUBROOT_FOLDERS:
        subroot_path = os.path.join(root, subroot)
        if not os.path.isdir(subroot_path):
            os.makedirs(subroot_path)


generate_resources_filetree()
