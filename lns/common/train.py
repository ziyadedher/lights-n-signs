"""Manages a common interface for training.

Provides simple functionality for easier implementation of new training
methods following in the spirit of streamlining the training and testing
process.
"""
from typing import (
    Generic, TypeVar, Type, Callable, Optional, Union, Dict, Tuple, NamedTuple
)

import os
import shutil
from enum import Enum

from lns.common import config
from lns.common.preprocess import Preprocessor
from lns.common.dataset import Dataset
from lns.common.process import ProcessedData, Processor
from lns.common.model import Model


ModelType = TypeVar("ModelType", bound=Model)
ProcessedDataType = TypeVar("ProcessedDataType", bound=ProcessedData)


class TrainerNotSetupException(Exception):
    """Raised when training is attemped to be started without setup."""


class Trainer(Generic[ModelType, ProcessedDataType]):
    """Abstract trainer class managing high level aspects of training."""

    model: Optional[ModelType]

    _paths: Dict[str, str]
    _data: ProcessedDataType

    __name: str
    __dataset: Dataset

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

    # TODO(ziyadedher): Might want to have the dictionary be a NamedTuple as well.
    def __init__(self, name: str, dataset: Union[str, Dataset], *,
                 _processor: Type[Processor[ProcessedDataType]], _method: str, _load: bool,
                 _subpaths: Dict[str, Subpath]) -> None:
        """Initialize a trainer.

        Generates a trainer with the given <name> on the given <dataset> which
        could be either a `Dataset` object or the name of a dataset.

        Needs some metadata to function correctly including the following:
        <_processor> is the specific processor class that is used for this method of training.
        <_method> is the unique name of the method we are training.
        <_load> determines whether to keep non-temporal folders and files.
        <_subpaths> is a dictionary of unique path name to `Subpath`.
        """
        self.model = None
        self._paths = {}
        self.__name = name

        # Get preprocess data if required
        if isinstance(dataset, str):
            self.__dataset = Preprocessor.preprocess(dataset)
        elif isinstance(dataset, Dataset):
            self.__dataset = dataset
        else:
            raise ValueError(f"<dataset> may only be `str` or `Dataset`, not {type(dataset)}")
        # Get processed data from the preprocessed dataset
        self._data = _processor.process(self.__dataset)

        # Find the training root folder with the trainer name
        self._generate_filestructure(_load, _method, _subpaths)

    @property
    def name(self) -> str:
        """Get the unique name of this training configuration."""
        return self.__name

    @property
    def dataset(self) -> Dataset:
        """Get the dataset that this trainer is operating on."""
        return self.__dataset

    def generate_model(self) -> Optional[ModelType]:
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
                    os.mknod(path)
                elif subpath.path_type == Trainer.PathType.FOLDER:
                    os.makedirs(path)
