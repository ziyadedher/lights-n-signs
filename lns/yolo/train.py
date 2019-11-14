"""YOLOv3 trainer.

The module manages the representation of a YOLOv3 training session along with all associated data.
"""
import dataclasses
import os
from typing import Optional, Union

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.train import Trainer
from lns.yolo.model import YoloModel
from lns.yolo.process import YoloData, YoloProcessor
from lns.yolo.settings import YoloSettings
from lns.yolo._lib.get_kmeans import get_kmeans


class YoloTrainer(Trainer[YoloModel, YoloData, YoloSettings]):
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
                         _processor=YoloProcessor, _settings=YoloSettings,
                         _load=load, _subpaths=YoloTrainer.SUBPATHS)

    @property
    def model(self) -> Optional[YoloModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        weights = self.get_weights_path()
        anchors = self._paths["anchors_file"]
        classes = self.data.get_classes()

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
        settings = settings if settings else self._load_settings()
        self._set_settings(settings)

        anchors = get_kmeans(self.data.get_annotations(), self.settings.num_clusters)
        anchor_string = ", ".join(f"{anchor[0]},{anchor[1]}" for anchor in anchors)
        with open(self._paths["anchor_file"], "w") as anchor_file:
            anchor_file.write(anchor_string)

        from lns.yolo._lib import args
        args.train_file = self.data.get_annotations()
        args.val_file = self.data.get_annotations()
        args.restore_path = self.settings.initial_weights if self.settings.initial_weights else self.get_weights_path()
        args.save_dir = self._paths["checkpoint_folder"] + "/"
        args.log_dir = self._paths["log_folder"]
        args.progress_log_path = self._paths["progress_file"]
        args.anchor_path = self._paths["anchors_file"]
        args.class_name_path = self.data.get_classes()
        for field, setting in dataclasses.asdict(self.settings).items():
            setattr(args, field, setting)
        args.init()

        # Importing train will begin training
        try:
            from lns.yolo._lib import train  # pylint:disable=unused-import  # noqa
        except KeyboardInterrupt:
            print(f"Training interrupted")
        else:
            print(f"Training completed succesfully")
