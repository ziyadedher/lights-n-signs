"""Squeezedet trainer.

The module manages the representation of a Squeezedet training session along with all associated data.
"""
import dataclasses
import os
from typing import Optional, Union

from lns.common import config
from lns.common.dataset import Dataset
from lns.common.train import Trainer
from lns.squeezedet.model import SqueezedetModel
from lns.squeezedet.process import SqueezedetData, SqueezedetProcessor
from lns.squeezedet.settings import SqueezedetSettings


class SqueezedetTrainer(Trainer[SqueezedetModel, SqueezedetData, SqueezedetSettings]):
    """Manages the Squeezedet training environment and execution.

    Contains and encapsulates all training setup and files under one namespace.
    """

    SUBPATHS = {
        "log_folder": Trainer.Subpath(
            path="log", temporal=True, required=True, path_type=Trainer.PathType.FOLDER),
        "anchors_file": Trainer.Subpath(
            path="anchors", temporal=False, required=False, path_type=Trainer.PathType.FILE),
        "config_file": Trainer.Subpath(
            path="config", temporal=True, required=False, path_type=Trainer.PathType.FILE),
    }

    INITIAL_WEIGHTS_NAME = "imagenet.h5"
    INITIAL_WEIGHTS = os.path.join(config.RESOURCES_ROOT, config.WEIGHTS_FOLDER_NAME, INITIAL_WEIGHTS_NAME)

    settings: SqueezedetSettings

    def __init__(self, name: str, dataset: Optional[Union[str, Dataset]] = None, load: bool = True) -> None:
        """Initialize a Squeezedet trainer with the given unique <name>.

        Sources data from the <dataset> given, if any.
        If <load> is set to False removes any existing training files before training.
        """
        super().__init__(name, dataset,
                         _processor=SqueezedetProcessor, _settings=SqueezedetSettings,
                         _load=load, _subpaths=SqueezedetTrainer.SUBPATHS)

        # TODO: dynamically generate k-means
        self._paths["anchors_file"] = os.path.join(config.RESOURCES_ROOT, config.WEIGHTS_FOLDER_NAME, "yolo_anchors")

    @property
    def model(self) -> Optional[SqueezedetModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        cfg = self._paths["config_file"]

        model = None
        if os.path.exists(cfg):
            model = SqueezedetModel(cfg, self.settings)
        return model

    def get_weights_path(self) -> str:
        """Get the path to most up-to-date weights associated with this trainer."""
        checkpoints_dir = os.path.join(self._paths["log_folder"], "checkpoints")
        if os.path.isdir(checkpoints_dir):
            checkpoints = os.listdir(checkpoints_dir)
            if checkpoints:
                return os.path.join(
                    checkpoints_dir,
                    max(checkpoints, key=lambda checkpoint: float(checkpoint.split(".")[1].split("-")[1])))
        return SqueezedetTrainer.INITIAL_WEIGHTS

    def train(self, settings: Optional[SqueezedetSettings] = None) -> None:
        """Begin training the model."""
        settings = settings if settings else self._load_settings()
        self._set_settings(settings)

        from lns.squeezedet._lib.config.create_config import squeezeDet_config
        cfg = squeezeDet_config("")
        cfg.CLASS_NAMES = self.dataset.classes
        cfg.CLASSES = len(cfg.CLASS_NAMES)
        cfg.CLASS_TO_IDX = dict(zip(cfg.CLASS_NAMES, range(cfg.CLASSES)))
        cfg.KEEP_PROB = 1 - self.settings.dropout_ratio
        for field, setting in dataclasses.asdict(self.settings).items():
            setattr(cfg, field.upper(), setting)
        cfg.ANCHOR_PER_GRID = len(cfg.ANCHOR_SEED)
        cfg.ANCHORS_WIDTH = cfg.IMAGE_WIDTH / self.settings.anchor_size
        cfg.ANCHORS_HEIGHT = cfg.IMAGE_HEIGHT / self.settings.anchor_size

        from lns.squeezedet._lib.config.create_config import save_dict
        save_dict(cfg, self._paths["config_file"])

        from lns.squeezedet._lib import train
        train.img_file = self.data.get_images()
        train.gt_file = self.data.get_labels()
        train.log_dir_name = self._paths["log_folder"]
        train.init_file = self.settings.initial_weights if self.settings.initial_weights else self.get_weights_path()
        train.EPOCHS = self.settings.num_epochs
        train.OPTIMIZER = self.settings.optimizer
        train.REDUCELRONPLATEAU = self.settings.reduce_lr_on_plateau
        train.VERBOSE = self.settings.verbose
        train.CONFIG = self._paths["config_file"]

        try:
            train.train()
        except KeyboardInterrupt:
            print(f"Training interrupted")
        else:
            print(f"Training completed succesfully")
