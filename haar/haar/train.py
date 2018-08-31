"""Haar cascade training script.

This script manages model training and generation.
"""
from typing import Optional, Dict

import os
import shutil
import subprocess

import cv2  # type: ignore

from common import config
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
    __dataset_name: str
    __paths: Dict[str, str]

    def __init__(self, name: str, dataset_name: str) -> None:
        """Initialize a trainer with the given unique <name>.

        Sources data from the dataset with the given <dataset_name> and raises
        `NoSuchDatasetException` if no such dataset exists.
        """
        self.model = None

        self._feature_size = -1
        self._light_type = None
        try:
            self._data = HaarProcessor.get_processed(self.__dataset_name)
        except config.NoSuchDatasetException as e:
            raise e

        self.__name = name
        self.__dataset_name = dataset_name

        __base = os.path.abspath(
            os.path.join(__file__, os.pardir, self.__name)
        )
        self.__paths = {
            "base_folder": __base,
            "vector_file": os.path.join(__base, "positive.vec"),
            "cascade_folder": os.path.join(__base, "cascade"),
            "cascade_file": os.path.join(__base, "cascade", "cascade.xml")
        }

        # Remove the training directory with this name if it exists
        # and generate a new one
        if os.path.isdir(self.__paths["base_folder"]):
            shutil.rmtree(self.__paths["base_folder"])
        os.mkdir(self.__paths["base_folder"])
        os.mkdir(self.__paths["cascade_folder"])

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
            raise TrainerNotSetupException

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
            HaarModel(
                cv2.CascadeClassifier(cascade_file),
                [self._light_type] if self._light_type is not None else []
            )
        else:
            self.model = None

        return self.model
