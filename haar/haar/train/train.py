"""Main training script.

This script manages model training and generation.
"""
from typing import Optional, List, Dict

import os
import shutil
import subprocess

from haar.common import config
from haar.common.model import Model
from haar.model import HaarModel

cv2 = config.get_opencv_bindings()


class TrainerNotSetupException(Exception):
    """Raised when training is attempted to be started without being set up."""

    pass


class Trainer:
    """Manages the training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    model: Optional[Model]

    _feature_size: int
    _light_types: List[str]

    __name: str
    __paths: Dict[str, str]

    def __init__(self, name: str) -> None:
        """Initialize a trainer with the given unique <name>."""
        self.model = None

        self._feature_size = -1
        self._light_types = []

        self.__name = name

        __base = os.path.abspath(os.path.join(__file__, "..", self.__name))
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
                       light_types: List[str]) -> None:
        """Generate and setup any files required for training.

        Create <num_samples> positive samples of the <light_types> with given
        <feature_size>.
        """
        vector_file = self.__paths["vector_file"]
        annotations_file = ""  # TODO: get annotations

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
        self._light_types = light_types

    def train(self, num_stages: int,
              num_positive: int, num_negative: int) -> None:
        """Begin training the model.

        Train for <num_stages> stages before automatically stopping and
        generating the trained model. Train on <num_positive> positive samples
        and <num_negative> negative samples.
        """
        if self._feature_size < 0:
            raise TrainerNotSetupException

        vector_file = self.__paths["vector_file"]
        cascade_folder = self.__paths["cascade_folder"]
        negative_annotations_file = ""  # TODO: get annotations
        feature_size = self._feature_size

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

    def generate_model(self) -> None:
        """Generate and store the currently available prediction model.

        Stored model may be `None` if there is no currently available model.
        """
        cascade_file = self.__paths["cascade_file"]

        if os.path.isfile(cascade_file):
            self.model = HaarModel(cv2.CascadeClassifier(cascade_file),
                                   self._light_types)
        else:
            self.model = None
