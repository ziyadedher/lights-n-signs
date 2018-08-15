"""Main training script.

This script manages model training and generation.
"""
from typing import Optional, List

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

    __is_setup: bool
    __name: str

    def __init__(self, name: str) -> None:
        """Initialize a trainer with the given unique <name>."""
        self.model = None

        self._feature_size = -1
        self._light_types = []

        self.__name = name
        self.__path = os.path.abspath(os.path.join(__file__, self.__name))
        self.__vector_path = os.path.join(self.__path, "positive.vec")
        self.__cascade_folder_path = os.path.join(self.__path, "cascade")

        # Remove the training directory with this name if it exists
        # and generate a new one
        if os.path.isdir(self.__path):
            shutil.rmtree(self.__path)
        os.mkdir(self.__path)
        os.mkdir(self.__cascade_folder_path)

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
        self._feature_size = feature_size
        annotations = ""  # TODO: get annotations

        subprocess.run(["opencv_createsamples",
                        "-info", str(annotations),
                        "-w", str(self._feature_size),
                        "-h", str(self._feature_size),
                        "-num", str(num_samples),
                        "-vec", str(self.__vector_path)])

    def train(self, num_stages: int,
              num_positive: int, num_negative: int) -> None:
        """Begin training the model.

        Train for <num_stages> stages before automatically stopping and
        generating the trained model. Train on <num_positive> positive samples
        and <num_negative> negative samples.
        """
        if self._feature_size < 0:
            raise TrainerNotSetupException
        negative_annotations = ""  # TODO: get annotations

        subprocess.run(["opencv_traincascade",
                        "-numPos", str(num_positive),
                        "-numNeg", str(num_negative),
                        "-numStages", str(num_stages),
                        "-vec", str(self.__vector_path),
                        "-bg", str(negative_annotations),
                        "-w", str(self._feature_size),
                        "-h", str(self._feature_size),
                        "-data", str(self.__cascade_folder_path)])

        self.generate_model()

    def generate_model(self) -> None:
        """Generate and store the currently available prediction model.

        Stored model may be `None` if there is no currently available model.
        """
        cascade_file = os.path.join(self.__cascade_folder_path, "cascade.xml")
        if os.path.isfile(cascade_file):
            self.model = HaarModel(cv2.CascadeClassifier(cascade_file))
        else:
            self.model = None
