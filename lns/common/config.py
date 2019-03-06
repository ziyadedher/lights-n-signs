"""General configuration file.

Use this file instead of hard-coding any directories or for any other general
configuration of the package.
"""
from typing import List

import os


RESOURCES_ROOT: str = os.path.abspath(
    # os.path.join(os.path.expanduser("~"), ".lns-training/resources")
    "/mnt/ssd1/lns/resources"
)

POSSIBLE_DATASETS: List[str] = [
    "LISA", "Bosch", "Custom", "Custom_testing", "sim", "mturk", "cities", "LISA_signs", "lights",
]

SEED = 6
MIN_SIZE = 40
