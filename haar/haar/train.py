"""Haar cascade training script.

This script manages model training and generation.
"""
from typing import Optional, Dict, Union

import os
import shutil
import subprocess

import cv2  # type: ignore

from common import config
from common.preprocess.preprocess import Preprocessor
from common.preprocess.preprocessing import Dataset
from haar.model import HaarModel
from haar.process import HaarData, HaarProcessor


class TrainerNotSetupException(Exception):
    """Raised when training is attempted to be started without being set up."""

    pass


class Trainer:
    """Manages the training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    model: Optional[HaarModel]

    _feature_size: int
    _light_type: Optional[str]
    _data: HaarData

    __name: str
    __dataset: Dataset
    __paths: Dict[str, str]

    def __init__(self, name: str,
                 dataset: Union[str, Dataset]) -> None:
        """Initialize a trainer with the given unique <name>.

        Sources data from the <dataset> given which can either be a name of
        an available dataset or a `PreprocessingData` object.
        """
        self.model = None
        self.__name = name

        if isinstance(dataset, str):
            self.__dataset = Preprocessor.preprocess(dataset)
        elif isinstance(dataset, Dataset):
            self.__dataset = dataset
        else:
            raise ValueError(
                "`dataset` may only be `str` or `PreprocessingData`, not" +
                f"{type(dataset)}"
            )

        self._feature_size = -1
        self._light_type = None
        self._data = HaarProcessor.process(self.__dataset)

        # Set up the required paths
        __trainer = os.path.join(
            config.RESOURCES_ROOT, "haar/trainers", self.__name
        )
        self.__paths = {
            "vector_file": os.path.join(__trainer, "positive.vec"),
            "cascade_folder": os.path.join(__trainer, "cascade"),
            "cascade_file": os.path.join(__trainer, "cascade", "cascade.xml")
        }

        # Remove the training directory with this name if it exists
        # and generate a new one
        if os.path.isdir(__trainer):
            shutil.rmtree(__trainer)
        os.makedirs(__trainer)
        os.makedirs(self.__paths["cascade_folder"])

    @property
    def name(self) -> str:
        """Get the unique name of this training configuration."""
        return self.__name

    def setup_training(self, feature_size: int, num_samples: int,
                       light_type: str) -> None:
        """Generate and setup any files required for training.

        Create <num_samples> positive samples of the <light_type> with given
        <feature_size>.
        """
        vector_file = self.__paths["vector_file"]
        try:
            annotations_file = self._data.get_positive_annotation(light_type)
        except KeyError:
            print("No positive annotations for light type " +
                  f"`{light_type}` available.")
            return

        command = [
            "opencv_createsamples",
            "-info", str(annotations_file),
            "-w", str(feature_size),
            "-h", str(feature_size),
            "-num", str(num_samples),
            "-vec", str(vector_file)
        ]
        subprocess.run(command)

        self._feature_size = feature_size
        self._light_type = light_type

    def train(self, num_stages: int,
              num_positive: int, num_negative: int) -> None:
        """Begin training the model.

        Train for <num_stages> stages before automatically stopping and
        generating the trained model. Train on <num_positive> positive samples
        and <num_negative> negative samples.
        """
        if self._feature_size < 0 or self._light_type is None:
            raise TrainerNotSetupException(
                "Trainer has not been set up using `setup_training`."
            )

        vector_file = self.__paths["vector_file"]
        cascade_folder = self.__paths["cascade_folder"]
        feature_size = self._feature_size
        try:
            negative_annotations_file = self._data.get_negative_annotation(
                self._light_type
            )
        except KeyError:
            print("No positive annotations for light type " +
                  f"`{self._light_type}` available.")
            return

        command = [
            "opencv_traincascade",
            "-numPos", str(num_positive),
            "-numNeg", str(num_negative),
            "-numStages", str(num_stages),
            "-vec", str(vector_file),
            "-bg", str(negative_annotations_file),
            "-w", str(feature_size),
            "-h", str(feature_size),
            "-data", str(cascade_folder)
        ]
        subprocess.run(command)

        self.generate_model()

    def generate_model(self) -> Optional[HaarModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        cascade_file = self.__paths["cascade_file"]

        if os.path.isfile(cascade_file):
            self.model = HaarModel(
                cv2.CascadeClassifier(cascade_file),
                [self._light_type] if self._light_type is not None else []
            )
        else:
            self.model = None

        return self.model
