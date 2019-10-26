"""Manages a common interface for training.

Provides simple functionality for easier implementation of new training
methods following in the spirit of streamlining the training and testing
process.
"""
from typing import (
    Generic, TypeVar, Type, Callable, Optional, Union, Dict, NamedTuple
)

import os
import shutil
import pickle
from enum import Enum

from lns.common import config
from lns.common.preprocess import Preprocessor
from lns.common.dataset import Dataset
from lns.common.process import ProcessedData, Processor
from lns.common.model import Model


ModelType = TypeVar("ModelType", bound=Model)
ProcessedDataType = TypeVar("ProcessedDataType", bound=ProcessedData)


class NoTrainerDataset(Exception):
    """Raised when trying to work with trainer data in a trainer with no dataset."""


class Trainer(Generic[ModelType, ProcessedDataType]):
    """Abstract trainer class managing high level aspects of training."""

    _paths: Dict[str, str]

    __name: str
    __data: Optional[ProcessedDataType]
    __dataset: Optional[Dataset]

    SetupFunc = TypeVar("SetupFunc", bound=Callable[..., None])
    TrainFunc = TypeVar("TrainFunc", bound=Callable[..., None])

    class PathType(Enum):
        """Represents the type of path in the subpaths dictionary."""

        FOLDER = 0
        FILE = 1

    class Subpath(NamedTuple):
        """Represents a single subpath in the trainer."""

        path: str
        temporal: bool
        required: bool
        path_type: 'Trainer.PathType'

    BUILTIN_SUBPATHS = {
        "_dataset": Subpath(
            path="dataset", temporal=False, required=False, path_type=PathType.FILE),
        "_data": Subpath(
            path="data", temporal=False, required=False, path_type=PathType.FILE),
    }

    def __init__(self, name: str, dataset: Optional[Union[str, Dataset]] = None, *,
                 _processor: Type[Processor[ProcessedDataType]], _load: bool, _subpaths: Dict[str, Subpath]) -> None:
        """Initialize a trainer.

        Generates a trainer with the given <name> on the given <dataset> if given.

        Needs some metadata to function correctly including the following:
        <_processor> is the specific processor class that is used for this method of training.
        <_load> determines whether to keep non-temporal folders and files.
        <_subpaths> is a dictionary of unique path name to `Subpath`.
        """
        self._paths = {}
        self.__name = name
        self.__data = None
        self.__dataset = None

        self._generate_filestructure(_load, _processor.method(), {**Trainer.BUILTIN_SUBPATHS, **_subpaths})
        self._acquire_data(dataset, _processor)

    @property
    def name(self) -> str:
        """Get the unique name of this training configuration."""
        return self.__name

    @property
    def dataset(self) -> Dataset:
        """Get the dataset that this trainer is operating on.

        Raises a `NoTrainerDataset` exception if no dataset has been provided.
        """
        if not self.__dataset:
            raise NoTrainerDataset("No dataset provided for this instance of the trainer.")
        return self.__dataset

    @property
    def data(self) -> ProcessedDataType:
        """Get the processed dataset that this trainer is operating on.

        Raises a `NoTrainerDataset` exception if no dataset has been provided.
        """
        if not self.__data:
            raise NoTrainerDataset("No dataset provided for this instance of the trainer.")
        return self.__data

    @property
    def model(self) -> Optional[ModelType]:
        """Generate and return the currently available model.

        Model may be `None` if there is no currently available model.
        """
        raise NotImplementedError

    def _generate_filestructure(self, load: bool, method: str, subpaths: Dict[str, Subpath]) -> None:
        trainer_root = os.path.join(
            config.RESOURCES_ROOT, config.TRAINERS_FOLDER_NAME, method, self.__name
        )
        self._paths["trainer_root"] = trainer_root

        for name, subpath in subpaths.items():
            # Generate absolute paths to each file we want to track
            path = os.path.join(trainer_root, subpath.path)
            self._paths[name] = path

            # Remove files and folders that we do not want to keep,
            # based on whether or not we are loading from previously trained
            if subpath.temporal or not load:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            if subpath.required and not os.path.exists(path):
                if subpath.path_type == Trainer.PathType.FILE:
                    directory = os.path.dirname(path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    open(path, 'a').close()
                elif subpath.path_type == Trainer.PathType.FOLDER:
                    os.makedirs(path)

    def _acquire_data(self, dataset: Optional[Union[Dataset, str]],
                      _processor: Type[Processor[ProcessedDataType]]) -> None:
        if dataset:
            # Get preprocess data if required
            if isinstance(dataset, str):
                self.__dataset = Preprocessor.preprocess(dataset)
            elif isinstance(dataset, Dataset):
                self.__dataset = dataset
            else:
                raise ValueError(f"<dataset> may only be `str` or `Dataset`, not {type(dataset)}")
            # Get processed data from the preprocessed dataset
            self.__data = _processor.process(self.__dataset)

            with open(self._paths["_dataset"], "wb") as dataset_file:
                pickle.dump(self.__dataset, dataset_file)
            with open(self._paths["_data"], "wb") as data_file:
                pickle.dump(self.__data, data_file)

        elif os.path.isfile(self._paths["_dataset"]) and os.path.isfile(self._paths["_data"]):
            with open(self._paths["_dataset"], "rb") as dataset_file:
                self.__dataset = pickle.load(dataset_file)
            with open(self._paths["_data"], "rb") as data_file:
                self.__data = pickle.load(data_file)
            print("Data loaded from trainer cache.")
