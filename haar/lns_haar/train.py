"""Haar cascade training script.

This script manages model training and generation.
"""
from typing import Optional, Union

import os
import subprocess

import cv2  # type: ignore

from lns_common.train import Trainer
from lns_common.preprocess.preprocessing import Dataset
from lns_haar.model import HaarModel
from lns_haar.process import HaarData, HaarProcessor
from haar.preprocessing.artificial import SyntheticDataset
from mergevec.mergevec import merge_vec_files


class HaarTrainer(Trainer[HaarModel, HaarData]):
    """Manages the training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    _feature_size: int
    _light_type: Optional[str]
    _is_synthetic: bool

    __subpaths = {
        "vector_file": (
            "positive.vec", True, False, "file"
        ),
        "cascade_folder": (
            "cascade", True, True, "folder"
        ),
        "cascade_file": (
            os.path.join("cascade", "cascade.xml"), True, False, "file"
        )
    }

    def __init__(self, name: str,
                 dataset: Union[str, Dataset],
                 load: bool = False) -> None:
        """Initialize a Haar trainer with the given unique <name>.

        Sources data from the <dataset> given which can either be a name of
        an available dataset or a `Dataset` object. If <load> is set
        to True attempts to load the trainer with the given ID before
        overwriting.
        """
        self._is_synthetic = isinstance(dataset, SyntheticDataset)

        super().__init__(name, dataset,
                         _processor=HaarProcessor, _type="haar", _load=load,
                         _subpaths=HaarTrainer.__subpaths)

        self._feature_size = -1
        self._light_type = None

    @Trainer._setup
    def setup_haar(self, feature_size: int, num_samples: int,
                   light_type: str) -> None:
        """Generate and setup any files required for training.

        Create <num_samples> positive samples of the <light_type> with given
        <feature_size>.
        """
        vector_file = self._paths["vector_file"]

        if self._is_synthetic:
            vecs_dir = os.path.join(
                self.__dataset.path_to_source, "vecs"  # type: ignore
            )
            augmented_samples = os.path.join(
                self.__dataset.path_to_source, "output"  # type: ignore
            )

            try:
                neg_annotations_file = self._data.get_negative_annotation(
                    str(self._light_type)
                )
            except KeyError:
                print("No negative annotations for light type " +
                      f"`{self._light_type}` available.")
                return

            os.mkdir(vecs_dir)
            subprocess.run(
                [
                    "./create_samples_multi.sh",
                    augmented_samples,
                    vecs_dir,
                    str(self.__dataset.samples_multiplier),  # type: ignore
                    str(neg_annotations_file)
                ]
            )
            merge_vec_files(vecs_dir, str(vector_file))
        else:
            try:
                pos_annotations_file = self._data.get_positive_annotation(
                    light_type
                )
            except KeyError:
                print("No positive annotations for light type " +
                      f"`{light_type}` available.")
                return

            command = [
                "opencv_createsamples",
                "-info", str(pos_annotations_file),
                "-w", str(feature_size),
                "-h", str(feature_size),
                "-num", str(num_samples),
                "-vec", str(vector_file)
            ]
            subprocess.run(command)

        self._light_type = light_type
        self._feature_size = feature_size

    @Trainer._train
    def train_haar(self, num_stages: int,
                   num_positive: int, num_negative: int) -> None:
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
            neg_annotations_file = self._data.get_negative_annotation(
                self._light_type
            )
        except KeyError:
            print("No negative annotations for light type " +
                  f"`{self._light_type}` available.")
            return

        command = [
            "opencv_traincascade",
            "-numPos", str(num_positive),
            "-numNeg", str(num_negative),
            "-numStages", str(num_stages),
            "-vec", str(vector_file),
            "-bg", str(neg_annotations_file),
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
                self.train_haar(stage + 1, num_positive, num_negative)
            else:
                print(f"Training ended prematurely, no stages were trained.")
        else:
            # Makes sure the cascade has been generated if
            # the training ended normally
            if "cascade.xml" in os.listdir(cascade_folder):
                print(f"Training completed at stage {num_stages - 1}.")
            else:
                print("Something went wrong, no cascade generated.")
            self.generate_model()

    def generate_model(self) -> Optional[HaarModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        cascade_file = self._paths["cascade_file"]

        if os.path.isfile(cascade_file):
            self.model = HaarModel(
                cv2.CascadeClassifier(cascade_file),
                [self._light_type] if self._light_type is not None else []
            )
        else:
            self.model = None

        return self.model
