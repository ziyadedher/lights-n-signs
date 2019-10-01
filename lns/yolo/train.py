"""YOLOv3 trainer.

The module manages the representation of a YOLOv3 training session along with all associated data.
"""
from typing import Optional, Union

import os

from lns.common import config
from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.yolo.model import YoloModel
from lns.yolo.process import YoloData, YoloProcessor


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
    }

    INITIAL_WEIGHTS_NAME = "yolo_coco"
    INITIAL_WEIGHTS = os.path.join(config.RESOURCES_ROOT, config.WEIGHTS_FOLDER_NAME, INITIAL_WEIGHTS_NAME)

    def __init__(self, name: str, dataset: Union[str, Dataset], load: bool = True) -> None:
        """Initialize a YOLOv3 trainer with the given unique <name>.

        Sources data from the <dataset> given which can either be a name of an available dataset or a `Dataset` object.
        If <load> is set to False removes any existing training files before training.
        """
        super().__init__(name, dataset,
                         _processor=YoloProcessor, _method=YoloProcessor.method(), _load=load,
                         _subpaths=YoloTrainer.SUBPATHS)

    def _get_initial_weight(self) -> str:
        checkpoints = os.listdir(self._paths["checkpoint_folder"])
        checkpoints.remove("checkpoint")
        if checkpoints:
            return os.path.join(self._paths["checkpoint_folder"],
                                max(checkpoints, key=lambda checkpoint: int(checkpoint.split("_")[1])))
        return YoloTrainer.INITIAL_WEIGHTS

    def train(self, initial_weight: Optional[str] = None) -> None:
        """Begin training the model."""
        # TODO: dynamically generate k-means
        self._paths["anchors_file"] = os.path.join(config.RESOURCES_ROOT, config.WEIGHTS_FOLDER_NAME, "yolo_anchors")

        from lns.yolo._lib import args
        args.train_file = self._data.get_annotations()
        args.val_file = self._data.get_annotations()
        args.restore_path = initial_weight if initial_weight else self._get_initial_weight()
        args.save_dir = self._paths["checkpoint_folder"]
        args.log_dir = self._paths["log_folder"]
        args.progress_log_path = self._paths["progress_file"]
        args.anchor_path = self._paths["anchors_file"]
        args.class_name_path = self._data.get_classes()

        # Importing train will begin training
        from lns.yolo._lib import train  # pylint:disable=unused-import  # noqa

    def generate_model(self) -> Optional[YoloModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        return self.model
