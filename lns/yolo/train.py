"""YOLOv3 trainer.

The module manages the representation of a YOLOv3 training session along with all associated data.
"""
from typing import Optional, Union

import os
import json

from lns.common import config
from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.yolo.model import YoloModel
from lns.yolo.process import YoloData, YoloProcessor
from lns.yolo.settings import YoloSettings


class YoloTrainer(Trainer[YoloModel, YoloData]):
    """Manages the YOLOv3 training environment and execution.

    Contains and encapsulates all training setup and files under one namespace.
    """

    SUBPATHS = {
        "log_folder": Trainer.Subpath(
            path="log", temporal=True, required=True, path_type=Trainer.PathType.FOLDER),
        "checkpoint_folder": Trainer.Subpath(
            path="checkpoint", temporal=False, required=True, path_type=Trainer.PathType.FOLDER),
        "anchors_file": Trainer.Subpath(
            path="anchors", temporal=False, required=False, path_type=Trainer.PathType.FILE),
        "progress_file": Trainer.Subpath(
            path="progress", temporal=True, required=False, path_type=Trainer.PathType.FILE),
        "settings_file": Trainer.Subpath(
            path="settings", temporal=False, required=False, path_type=Trainer.PathType.FILE),
    }

    INITIAL_WEIGHTS_NAME = "yolov3.ckpt"
    INITIAL_WEIGHTS = os.path.join(config.RESOURCES_ROOT, config.WEIGHTS_FOLDER_NAME, INITIAL_WEIGHTS_NAME)

    settings: YoloSettings

    def __init__(self, name: str, dataset: Optional[Union[str, Dataset]] = None, load: bool = True) -> None:
        """Initialize a YOLOv3 trainer with the given unique <name>.

        Sources data from the <dataset> given, if any.
        If <load> is set to False removes any existing training files before training.
        """
        super().__init__(name, dataset,
                         _processor=YoloProcessor, _load=load, _subpaths=YoloTrainer.SUBPATHS)

        self.settings = self._load_settings()
        # TODO: dynamically generate k-means
        self._paths["anchors_file"] = os.path.join(config.RESOURCES_ROOT, config.WEIGHTS_FOLDER_NAME, "yolo_anchors")

    @property
    def model(self) -> Optional[YoloModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        weights = self.get_weights_path()
        anchors = self._paths["anchors_file"]
        classes = self.settings.class_name_path

        model = None
        if all(os.path.exists(path) for path in (anchors, classes)):
            model = YoloModel(weights, anchors, classes, self.settings)
        return model

    def get_weights_path(self) -> str:
        """Get the path to most up-to-date weights associated with this trainer."""
        checkpoints = os.listdir(self._paths["checkpoint_folder"])
        if "checkpoint" in checkpoints:
            checkpoints.remove("checkpoint")
        if checkpoints:
            return os.path.join(
                self._paths["checkpoint_folder"],
                max(checkpoints, key=lambda checkpoint: int(checkpoint.split("_")[3])).rsplit(".", 1)[0])
        return YoloTrainer.INITIAL_WEIGHTS

    def train(self, settings: Optional[YoloSettings] = None) -> None:
        """Begin training the model."""
        self.settings = settings if settings else self._load_settings()
        self.settings._replace(
            train_file=self.data.get_annotations(),
            val_file=self.data.get_annotations(),
            restore_path=self.settings.initial_weights if self.settings.initial_weights else self.get_weights_path(),
            save_dir=self._paths["checkpoint_folder"] + "/",
            log_dir=self._paths["log_folder"],
            progress_log_path=self._paths["progress_file"],
            anchor_path=self._paths["anchors_file"],
            class_name_path=self.data.get_classes(),
        )

        with open(self._paths["settings_file"], "w") as file:
            json.dump(self.settings._asdict(), file)

        from lns.yolo._lib import args
        for field, setting in zip(self.settings._fields, self.settings):
            setattr(args, field, setting)
        args.init()

        # Importing train will begin training
        try:
            from lns.yolo._lib import train  # pylint:disable=unused-import  # noqa
        except KeyboardInterrupt:
            print(f"Training interrupted")
        else:
            print(f"Training completed succesfully")

    def _load_settings(self) -> YoloSettings:
        settings = YoloSettings()
        if os.path.exists(self._paths["settings_file"]):
            with open(self._paths["settings_file"], "r") as file:
                settings = YoloSettings(**json.load(file))
        return settings
