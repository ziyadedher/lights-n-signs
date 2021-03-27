"""Haar cascade training script.

This script manages model training and generation.
"""
from typing import Optional, Union

import os
import subprocess
import dataclasses

from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.haar.model import HaarModel
from lns.haar.process import HaarData, HaarProcessor
from lns.haar.settings import HaarSettings

class HaarTrainer(Trainer[HaarModel, HaarData, HaarSettings]):
    """Manages the training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    SUBPATHS = {
        "vector_file": Trainer.Subpath(
            path="positive.vec", temporal=False, required=False, path_type=Trainer.PathType.FILE),
        "cascade_folder": Trainer.Subpath(
            path="cascade", temporal=False, required=True, path_type=Trainer.PathType.FOLDER),
        "cascade_file": Trainer.Subpath(
            path="cascade/cascade.xml", temporal=False, required=True, path_type=Trainer.PathType.FILE),
    }

    def __init__(self, name: str, class_index: int, dataset: Optional[Union[str, Dataset]] = None, load: bool = True, forcePreprocessing=False) -> None:
        """Initialize a Haar trainer with the given unique <name>.

        Sources data from the given <dataset>, if any.
        If <load> is set to False removes any existing trained cascade files before training.
        """
        super().__init__(name, dataset,
                         _processor=HaarProcessor, _settings=HaarSettings,
                         _load=load, _subpaths=HaarTrainer.SUBPATHS, forcePreprocessing=forcePreprocessing)
        self.class_index = class_index
        print("Class: " + str(dataset.classes[class_index]))

    @property
    def model(self) -> Optional[HaarModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        cascade_file = self._paths["cascade_file"]

        model = None
        if os.path.isfile(cascade_file) and self.class_index is not None:
            model = HaarModel(cascade_file, self.settings)
        return model

    def setup(self, settings: Optional[HaarSettings] = None) -> None:
        """Generate and setup any files required for training.

        Create <num_samples> positive samples of the class represented by the <class_index> with given <feature_size>.
        """
        print("Setting up...")
        settings = settings if settings else self._load_settings()
        self._set_settings(settings)

        vector_file = self._paths["vector_file"]
        try:
            annotations_file = self.data.get_positive_annotation(self.class_index)
        except IndexError:
            print(f"No positive annotations for class index `{self.class_index}` available.")
            return

        print("\n\nSetup")
        command = [
            "/usr/bin/opencv_createsamples",
            "-info", str(annotations_file),
            "-w", str(settings.width),
            "-h", str(settings.height),
            "-num", str(settings.num_samples),
            "-vec", str(vector_file)
        ]
        print("vector file: " + str(vector_file))
        
        subprocess.run(command, check=False)

    def train(self, settings: Optional[HaarSettings] = None) -> None:
        """Begin training the model.

        Train for <num_stages> stages before automatically stopping and generating the trained model.
        Train on <num_positive> positive samples and <num_negative> negative samples.
        """
        settings = settings if settings else self._load_settings()
        self._set_settings(settings)

        vector_file = self._paths["vector_file"]
        cascade_folder = self._paths["cascade_folder"]
        try:
            negative_annotations_file = self.data.get_negative_annotation(self.class_index)
        except KeyError:
            print(f"No negative annotations for class index `{self.class_index}` available.")
            return

        # Hack to get around issue with opencv_traincascade needing relative path for `-bg`
        os.chdir(os.path.dirname(negative_annotations_file))
        negative_annotations_file = os.path.basename(negative_annotations_file)
        print('\n\n')
        print("Negative annotations: " + str(negative_annotations_file))
        print("Vector File: " + str(vector_file))
        print("Training model for: " + self.dataset.classes[self.class_index])

        print("\n\nTraining")
        command = [
            "/usr/bin/opencv_traincascade",
            "-featureType", str(self.settings.feature_type),
            "-numPos", str(self.settings.num_positive),
            "-numNeg", str(self.settings.num_negative),
            "-numStages", str(self.settings.num_stages),
            "-minHitRate", str(self.settings.min_hit_rate),
            "-maxFalseAlarmRate", str(self.settings.max_false_alarm),
            "-vec", str(vector_file),
            "-bg", str(negative_annotations_file),
            "-w", str(self.settings.width),
            "-h", str(self.settings.height),
            "-data", str(cascade_folder),
            "-precalcValBufSize", str(256)
        ]

        try:
            subprocess.run(command, check=False)
        except KeyboardInterrupt:
            # Find the highest stage that has been trained
            stage = max(
                int(file_name[5:-4])  # Grab the number out of "stageXXX.xml"
                for file_name in os.listdir(cascade_folder)
                if file_name.startswith("stage") and file_name.endswith(".xml")
            ) or -1

            if stage > -1:
                self.train(dataclasses.replace(self.settings, num_stages=stage + 1))
            else:
                print("Training ended prematurely, no stages were trained.")
        else:
            # Makes sure the cascade has been generated if the training ended normally
            if "cascade.xml" in os.listdir(cascade_folder):
                print(f"Training completed at stage {self.settings.num_stages - 1}.")
            else:
                print("Something went wrong, no cascade generated.")
