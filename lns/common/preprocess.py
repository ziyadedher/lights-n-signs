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
                "x_min": x_coordinate_of_top_left_corner,
                "y_min": y_coordinate_of_top_left_corner,
                "x_max": x_coordinate_of_bottom_right_corner,
                "y_max": y_coordinate_of_bottom_right_corner
            },
            ...
        ],
        ...
    }
}
"""
from typing import Dict, Callable

from lns.common.dataset import Data, Dataset


DatasetPreprocessor = Callable[[str], Dataset]


class Preprocessor:
    """Manager for all preprocessing and retrieval of preprocessed data."""

    _dataset_preprocessors: Dict[str, DatasetPreprocessor] = {}
    _preprocessing_data: Dict[str, Dataset] = {}

    @classmethod
    def register_default_preprocessors(cls) -> None:
        """Register default preprocessors in preprocessing."""
        from lns.common import preprocessing  # noqa: needed to register all the preprocessing functions

    @classmethod
    def register_dataset_preprocessor(cls, name: str) -> Callable[[DatasetPreprocessor], DatasetPreprocessor]:
        """Register a dataset preprocessor with a given name."""
        def _register_dataset_preprocessor(preprocessor: DatasetPreprocessor) -> DatasetPreprocessor:
            cls._dataset_preprocessors[name] = preprocessor
            return preprocessor
        return _register_dataset_preprocessor

    @classmethod
    def preprocess(cls, dataset_name: str, force: bool = False, scale: float = 1.0) -> Dataset:
        """Preprocess the dataset with the given name and return the result.

        Setting <force> to `True` will force a preprocessing even if the
        preprocessed data already exists in memory. Setting <scale> to be
        anything other than `1.0` will force a dataset copying and
        reconstruction with the given scale.

        Raises `ValueError` if a preprocessor for the dataset does
        not exist.
        """
        # Uses memoization to speed up preprocessing acquisition
        if not force and scale == 1.0 and dataset_name in cls._preprocessing_data:
            return cls._preprocessing_data[dataset_name]

        try:
            dataset_path = Data.get_dataset_path(dataset_name)
        except KeyError:
            raise ValueError(f"No such dataset {dataset_name} available. Did you add it to `config`?")

        try:
            _preprocessor = cls._dataset_preprocessors[dataset_name]
        except KeyError:
            raise ValueError(f"Dataset {dataset_name} has no allocated preprocessor. Did you register it?")

        dataset = _preprocessor(dataset_path)
        cls._preprocessing_data[dataset_name] = dataset
        if scale != 1.0:
            dataset = dataset.scale(scale)

        return dataset
