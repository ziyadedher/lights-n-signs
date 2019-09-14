"""Abstract data processing for training methods.

Provides an interface for data processing for all detection methods to make
implementation of new detection methods easier and more streamlined.
"""
from typing import Dict, TypeVar, Generic

import os
import shutil
import pickle

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.resources import Resources


class ProcessedData:
    """Abstract data container for data after processing."""


ProcessedDataType = TypeVar("ProcessedDataType", bound=ProcessedData)


class Processor(Generic[ProcessedDataType]):
    """Abstract processor for generation of processed data."""

    _processed_data: Dict[str, ProcessedDataType] = {}

    @classmethod
    def method(cls) -> str:
        """Get the training method this processor is for."""
        raise NotImplementedError

    @classmethod
    def get_processed_data_path(cls) -> str:
        """Get the folder with all the processed datasets for this processor."""
        return os.path.join(Resources.get_root(), config.PROCESSED_DATA_FOLDER_NAME, cls.method())

    @classmethod
    def init_cached_processed_data(cls) -> None:
        """Initialize the processor with previously cached processed data from disk."""
        processed_data_path = cls.get_processed_data_path()
        for processed_data_pkl in os.listdir(processed_data_path):
            name = processed_data_pkl.strip(config.PKL_EXTENSION)
            with open(os.path.join(processed_data_path, processed_data_pkl), "rb") as file:
                processed_data = pickle.load(file)
            cls._processed_data[name] = processed_data

    @classmethod
    def cache_processed_data(cls, name: str, processed_data: ProcessedDataType) -> None:
        """Cache the given <processed_data> with the given <name>.

        Stores the processed data locally in memory but also on disk for persistence.
        """
        cls._processed_data[name] = processed_data
        processed_data_pkl = os.path.join(cls.get_processed_data_path(), name + config.PKL_EXTENSION)
        with open(processed_data_pkl, "wb") as file:
            pickle.dump(processed_data, file)

    @classmethod
    def process(cls, dataset: Dataset, *, force: bool = False) -> ProcessedDataType:
        """Process all required data from the given <dataset>.

        Generates a processed data object and returns it.

        If <force> is set to `True` then the method will force a processing of
        the dataset even if previous data have been cached.

        Raises `NoPreprocessorException` if a preprocessor for the given
        <dataset> does not exist.
        """
        # Uses memoization to speed up processing acquisition
        if not force and dataset.name in cls._processed_data:
            print(f"Getting dataset {dataset.name} from processed dataset cache.")
            return cls._processed_data[dataset.name]

        processed_data_path = os.path.join(cls.get_processed_data_path(), dataset.name)
        if os.path.exists(processed_data_path):
            shutil.rmtree(processed_data_path)
        os.makedirs(processed_data_path)
        processed_data: ProcessedDataType = cls._process(dataset)

        cls.cache_processed_data(dataset.name, processed_data)
        return processed_data

    @classmethod
    def _process(cls, dataset: Dataset) -> ProcessedDataType:
        raise NotImplementedError
