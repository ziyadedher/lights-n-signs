"""Abstract data processing for training methods.

Provides an interface for data processing for all detection methods to make
implementation of new detection methods easier and more streamlined.
"""
from typing import TypeVar, Generic

from lns_common.preprocess.preprocess import Dataset


class ProcessedData:
    """Abstract data container for data after processing."""

    pass


ProcessedDataType = TypeVar("ProcessedDataType", bound=ProcessedData)


class Processor(Generic[ProcessedDataType]):
    """Abstract processor for generation of processed data."""

    @classmethod
    def process(cls, dataset: Dataset, *,
                force: bool = False) -> ProcessedDataType:
        """Process all required data from the given <dataset>.

        Generates a processed data object and returns itself.

        If <force> is set to `True` then the method will force a processing of
        the dataset even if previous data have been cached.

        Raises `NoPreprocessorException` if a preprocessor for the given
        <dataset> does not exist.
        """
        raise NotImplementedError
