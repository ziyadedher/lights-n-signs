"""Manages a common interface for training.

Provides simple functionality for easier implementation of new training
methods following in the spirit of streamlining the training and testing
process.
"""
from typing import (
    Generic, TypeVar, Type, Callable, Optional, Union, Any, Dict, Tuple, cast
)

import os
import shutil

from lns_common import config
from lns_common.model import ModelType
from lns_common.process import ProcessedDataType, Processor
from lns_common.preprocess.preprocess import Preprocessor, Dataset
from haar.preprocessing.artificial import SyntheticDataset


class TrainerNotSetupException(Exception):
    """Raised when training is attemped to be started without setup."""

    pass


class Trainer(Generic[ModelType, ProcessedDataType]):
    """Abstract trainer class managing high level aspects of training."""

    model: Optional[ModelType]

    _paths: Dict[str, str]
    _data: ProcessedDataType

    __name: str
    __dataset: Dataset
    __is_setup: bool

    SetupFunc = TypeVar("SetupFunc", bound=Callable[..., Any])  # type: ignore
    TrainFunc = TypeVar("TrainFunc", bound=Callable[..., Any])  # type: ignore

    def __init__(self,
                 name: str, dataset: Union[str, Dataset, SyntheticDataset], *,
                 _processor: Type[Processor[ProcessedDataType]],
                 _type: str, _load: bool,
                 _subpaths: Dict[str, Tuple[str, bool, bool, str]]) -> None:
        """Initialize a trainer.

        Generates a trainer with the given <name> on the given <dataset> which
        could be either a `Dataset` object or a string represented a dataset
        name.

        Needs some metadata to function correctly including the following:
        <_processor> is the specific processor class that is used for this
        method of training. <_type> is the unique name of the type of
        classifier we are training. <_load> determines whether to keep the
        folders and files that are marked as able to be kept in the next
        argument. <_subpaths> is a dictionary of unique path name to path
        description; the path description is a quadruple of relative path
        of file or folder, whether or not this file or folder should be
        preserved if <_load> is set to True, whether or not this file or folder
        should be regenerated if it does not exist, and the type of path
        which could be either "file" or "folder".
        """
        self.model = None
        self._data = None  # type: ignore
        self._paths = {}
        self.__name = name
        self.__is_setup = False

        # Get preprocess data if required
        if isinstance(dataset, str):
            self.__dataset = Preprocessor.preprocess(dataset)
        elif isinstance(dataset, Dataset) or \
                isinstance(dataset, SyntheticDataset):
            self.__dataset = dataset
        else:
            raise ValueError(
                "`dataset` may only be `str` or `Dataset`, not" +
                f"{type(dataset)}"
            )

        if isinstance(self.__dataset, Dataset):
            # Get processed data from the preprocessed dataset
            self._data = _processor.process(self.__dataset)

            # Find the training root folder with the trainer name
            self._generate_filestructure(_load, _type, _subpaths)

    @property
    def name(self) -> str:
        """Get the unique name of this training configuration."""
        return self.__name

    @property
    def dataset(self) -> Dataset:
        """Get the dataset being used with this training configuration."""
        return self.__dataset

    @classmethod
    def _setup(cls, setup_call: SetupFunc) -> SetupFunc:
        """Set up the trainer for training."""
        def _setup_wrapper(*args, **kwargs):  # type: ignore
            setup_call(*args, **kwargs)
            args[0].__is_setup = True

        return cast(Trainer.SetupFunc, _setup_wrapper)

    @classmethod
    def _train(cls, train_call: TrainFunc) -> TrainFunc:
        """Begin training the model."""
        def _train_wrapper(*args, **kwargs):  # type: ignore
            if not args[0].__is_setup:
                raise TrainerNotSetupException(
                    f"Trainer has not been set up yet."
                )
            train_call(*args, **kwargs)
        return cast(Trainer.TrainFunc, _train_wrapper)

    def generate_model(self) -> Optional[ModelType]:
        """Generate and return the currently available model.

        Model may be `None` if there is no currently available model.
        """
        raise NotImplementedError

    def _generate_filestructure(self, _load: bool, _type: str,
                                _subpaths: Dict[str,
                                                Tuple[str, bool, bool, str]]
                                ) -> None:

        __trainer_root = os.path.join(
            config.RESOURCES_ROOT, f"{_type}", "trainers", self.__name
        )
        self._paths["trainer_root"] = __trainer_root

        for name, (subpath, keep, generate, type) in _subpaths.items():
            # Generate absolute paths to each file we want to track
            if type not in ("file", "folder"):
                continue
            path = os.path.join(self._paths["trainer_root"], subpath)
            self._paths[name] = path

            # Remove files and folders that we do not want to keep,
            # based on whether or not we are loading from previously trained
            if not _load or not keep:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            if not os.path.exists(path) and generate:
                if type == "file":
                    directory = os.path.dirname(path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    os.mknod(path)
                elif type == "folder":
                    os.makedirs(path)
