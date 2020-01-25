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
            object1: utils.Object2D
            object2: utils.Object2D
            ...
        ],
        ...
    }
}
"""
import copy
import importlib
import os
import pickle
import pkgutil
from typing import Callable, Dict

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.resources import Resources

DatasetPreprocessor = Callable[[str], Dataset]


class Preprocessor:
    """Manager for all preprocessing and retrieval of preprocessed data."""

    _dataset_preprocessors: Dict[str, DatasetPreprocessor] = {}
    _preprocessed_data: Dict[str, Dataset] = {}

    @classmethod
    def init_cached_preprocessed_data(cls) -> None:
        """Initialize the preprocessor with previously cached preprocessed data from disk."""
        preprocessed_data_path = os.path.join(Resources.get_root(), config.PREPROCESSED_DATA_FOLDER_NAME)
        for dataset_pkl in os.listdir(preprocessed_data_path):
            if not dataset_pkl.endswith(config.PKL_EXTENSION):
                continue
            name = dataset_pkl.strip(config.PKL_EXTENSION)
            try:
                with open(os.path.join(preprocessed_data_path, dataset_pkl), "rb") as file:
                    dataset = pickle.load(file)
            except pickle.PickleError:
                print(f"Something went wrong while reading `{name}` from preprocessor cache, skipping.")
            else:
                cls._preprocessed_data[name] = dataset

    @classmethod
    def cache_preprocessed_data(cls, name: str, dataset: Dataset) -> None:
        """Cache the given <dataset> with the given <name>.

        Stores the dataset locally in memory but also on disk for persistence.
        """
        cls._preprocessed_data[name] = dataset
        dataset_pkl = os.path.join(
            Resources.get_root(), config.PREPROCESSED_DATA_FOLDER_NAME, name + config.PKL_EXTENSION)
        with open(dataset_pkl, "wb") as file:
            pickle.dump(dataset, file)

        #TEST
        k = 5
        dataset.generate_anchors(k)
        #####

    @classmethod
    def register_default_preprocessors(cls) -> None:
        """Register default preprocessors in preprocessing."""
        # Import all preprocessing modules so that they are registered in the preprocessor
        from lns.common import preprocessing
        for _, name, _ in pkgutil.walk_packages(preprocessing.__path__):  # type: ignore
            importlib.import_module(preprocessing.__name__ + '.' + name)

    @classmethod
    def register_dataset_preprocessor(cls, name: str) -> Callable[[DatasetPreprocessor], DatasetPreprocessor]:
        """Register a dataset preprocessor with a given name."""
        def _register_dataset_preprocessor(preprocessor: DatasetPreprocessor) -> DatasetPreprocessor:
            cls._dataset_preprocessors[name] = preprocessor
            return preprocessor
        return _register_dataset_preprocessor

    @classmethod
    def preprocess(cls, name: str, force: bool = False, scale: float = 1.0) -> Dataset:
        """Preprocess the dataset with the given <name> and return the result.

        Setting <force> to `True` will force a preprocessing even if the
        preprocessed data already exists in memory. Setting <scale> to be
        anything other than `1.0` will force a dataset copying and
        reconstruction with the given scale.

        Raises `ValueError` if a preprocessor for the dataset does
        not exist.
        """
        if scale != 1.0:
            name = f"{name}-{scale}"
            raise NotImplementedError("Scale is not yet implemented.")

        # Uses memoization to speed up preprocessing acquisition
        if not force and name in cls._preprocessed_data:
            print(f"Getting dataset {name} from dataset cache.")
            return copy.copy(cls._preprocessed_data[name])

        try:
            dataset_path = Resources.get_dataset_path(name)
        except KeyError:
            raise ValueError(f"No such dataset {name} available. Did you add it to `config`?")

        try:
            _preprocessor = cls._dataset_preprocessors[name]
        except KeyError:
            raise ValueError(f"Dataset {name} has no allocated preprocessor. Did you register it?")

        dataset = _preprocessor(dataset_path)
        cls.cache_preprocessed_data(name, dataset)
        return copy.copy(dataset)


Preprocessor.init_cached_preprocessed_data()
Preprocessor.register_default_preprocessors()
