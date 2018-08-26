"""Common preprocessing step for available datasets.

Manages the generation of a JSON file corresponding to each datasets that
contains an intermediate form of the data annotations for easier use within
the various subpackages representing each method of detection.

Each dataset's generated file's structure will be very similar to encourage
generalization of training methods to multiple datasets. A summary of the
structure of the generated file is below.

{
    "images": [
        "/absolute/path/to/image.png",
        "/absolute/path/to/other_image.png",
        "/absolute/path/to/nested/image.png",
        ...
    ],

    "classes": [
        "class_one",
        "other_detection_class",
        ...
    }

    "annotations": {
        "/absolute/path/to/image.png": [
            {
                "class": index_of_class,
                "x": x_coordinate_of_left_side_of_bound,
                "y": y_coordinate_of_top_of_bound,
                "w": width_of_bound,
                "h": height_of_bound
            },
            ...
        ],
        ...
    }
}
"""
from typing import Dict

from common.config import NoSuchDatasetException, Data
from common.preprocess import preprocessing
from common.preprocess.preprocessing import PreprocessingData


class Preprocessor:
    """Manager for all preprocessing and retrieval of preprocessed data."""

    # Preprocessor allocation for each dataset
    _PREPROCESSORS = {
        "LISA": preprocessing.preprocess_LISA
    }

    _preprocessing_data: Dict[str, PreprocessingData] = {}

    @classmethod
    def preprocess(cls, dataset_name: str,
                   force: bool = False) -> PreprocessingData:
        """Preprocess the dataset with the given name and return the result.

        Setting <force> to `True` will force a preprocessing even if the
        preprocessed data already exists in memory.

        Raises `NoSuchDatasetException` if such a dataset does not exist.
        Raises `NoPreprocessorException` if a preprocessor for the dataset does
        not exist.
        """
        # Uses memoization to speed up preprocessing acquisition
        if not force and dataset_name in cls._preprocessing_data:
            return cls._preprocessing_data[dataset_name]

        try:
            dataset_path = Data.get_dataset_path(dataset_name)
        except NoSuchDatasetException as e:
            raise e

        try:
            _preprocessor = cls._PREPROCESSORS[dataset_name]
        except KeyError:
            raise NoPreprocessorException(
                "Dataset {} has no allocated preprocessor."
            )

        preprocessed = _preprocessor(dataset_path)
        cls._preprocessing_data[dataset_name] = preprocessed
        return preprocessed


class NoPreprocessorException(Exception):
    """Raised when a dataset does not have an allocated preprocessor."""

    pass
