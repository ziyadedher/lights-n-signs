"""Haar cascade training script.

This script manages model training and generation.
"""
from typing import Optional, Union

import os
import subprocess

import cv2  # type: ignore

from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.haar.model import HaarModel
from lns.haar.process import HaarData, HaarProcessor


class HaarTrainer(Trainer[HaarModel, HaarData]):
    """Manages the training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    _feature_size: Optional[int]
    _class_index: Optional[str]

    SUBPATHS = {
        "vector_file": Subpath(
            path="positive.vec", temporal=False, required=False, Trainer.PathType.FILE),
        "cascade_folder": Subpath(
            path="cascade", temporal=False, required=True, Trainer.PathType.FOLDER),
        "cascade_file": Subpath(
            path=os.path.join("cascade", "cascade.xml"), temoral=False, required=True, Trainer.PathType.FILE),
    }

    def __init__(self, name: str, dataset: Union[str, Dataset], load: bool = True) -> None:
        """Initialize a Haar trainer with the given unique <name>.

        Sources data from the <dataset> given which can either be a name of
        an available dataset or a `Dataset` object.  If <load> is set
        to False removes any existing trained cascade files before training.
        """
        super().__init__(name, dataset,
                         _processor=HaarProcessor, _type="haar", _load=load,
                         _subpaths=HaarTrainer.SUBPATHS)

        self._feature_size = None
        self._class_index = None

    def setup(self, feature_size: int, num_samples: int, class_index: int) -> None:
        """Generate and setup any files required for training.

        Create <num_samples> positive samples of the class represented
        by the <class_index> with given <feature_size>.
        """
        vector_file = self._paths["vector_file"]
        try:
            annotations_file = self._data.get_positive_annotation(class_index)
        except IndexError:
            print(f"No positive annotations for class index `{class_index}` available.")
            return

        self._feature_size = feature_size
        self._class_index = class_index

        command = [
            "opencv_createsamples",
            "-info", str(annotations_file),
            "-w", str(feature_size),
            "-h", str(feature_size),
            "-num", str(num_samples),
            "-vec", str(vector_file)
        ]
        subprocess.run(command)

    def train(self, num_stages: int, num_positive: int, num_negative: int) -> None:
        """Begin training the model.

        Train for <num_stages> stages before automatically stopping and
        generating the trained model. Train on <num_positive> positive samples
        and <num_negative> negative samples.
        """
        assert self._light_type is not None

        vector_file = self._paths["vector_file"]
        cascade_folder = self._paths["cascade_folder"]
        feature_size = self._feature_size
        try:
            negative_annotations_file = self._data.get_negative_annotation(self._class_index)
        except KeyError:
            print(f"No negative annotations for class index `{self._class_index}` available.")
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

        try:
            subprocess.run(command)
        except KeyboardInterrupt:
            # Find the highest stage that has been trained
            stage = max(
                int(file_name[5:-4])  # Grab the number out of "stageXXX.xml"
                for file_name in os.listdir(cascade_folder)
                if file_name.startswith("stage") and file_name.endswith(".xml")
            ) or -1

            if stage > -1:
                self.train(stage + 1, num_positive, num_negative)
            else:
                print(f"Training ended prematurely, no stages were trained.")
        else:
            # Makes sure the cascade has been generated if the training ended normally
            if "cascade.xml" in os.listdir(cascade_folder):
                print(f"Training completed at stage {num_stages - 1}.")
            else:
                print("Something went wrong, no cascade generated.")
        finally:
            self.generate_model()

    def generate_model(self) -> Optional[HaarModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        cascade_file = self._paths["cascade_file"]

        if os.path.isfile(cascade_file):
            self.model = HaarModel(cascade_file, self._class_index)
        else:
            self.model = None

        return self.model
