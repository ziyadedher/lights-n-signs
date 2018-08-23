"""Data processing for Haar cascade training.

Manages all data processing for the generation of data ready to be trained
on with OpenCV Haar training scripts.
"""
from typing import Dict


class HaarData:
    """Data container for all Haar processed data.

    Contains positive annotations for each type of light and negative
    annotations for each type of light as well from the dataset
    """

    __positive_annotations: Dict[str, str]
    __negative_annotations: Dict[str, str]

    def __init__(self,
                 positive_annotations: Dict[str, str],
                 negative_annotations: Dict[str, str]) -> None:
        """Initialize the data structure."""
        self.__positive_annotations = positive_annotations
        self.__negative_annotations = negative_annotations

    def get_positive_annotation(self, light_type: str) -> str:
        """Get the path to a positive annotation file for the given light type.

        Raises `KeyError` if no such light type is available.
        """
        try:
            return self.__positive_annotations[light_type]
        except KeyError as e:
            raise e

    def get_negative_annotation(self, light_type: str) -> str:
        """Get the path to a negative annotation file for the given light type.

        Raises `KeyError` if no such light type is available.
        """
        try:
            return self.__negative_annotations[light_type]
        except KeyError as e:
            raise e


class HaarProcessor:
    """Haar processor responsible for data processing to Haar-valid formats."""

    @classmethod
    def process(cls, dataset_name: str) -> HaarData:
        """Process all required data from the dataset with the given name.

        Returns and stores the data.

        Raises `NoSuchDatasetException` if such a dataset does not exist.
        """
        # TODO: implement
        raise NotImplementedError

    @classmethod
    def generate_annotations(cls) -> None:
        """Generate all annotation files needed for Haar training."""
        # TODO: implement
        raise NotImplementedError

    @classmethod
    def get_processed(cls, dataset_name: str) -> HaarData:
        """Get Haar processed data for the dataset with the given name.

        Process and return data if the dataset exists but has not been
        processed yet.

        Raises `NoSuchDatasetException` if such a dataset does not exist.
        """
        # TODO: implement
        raise NotImplementedError
