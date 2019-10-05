"""YOLOv3 trainer.

The module manages the representation of a YOLOv3 training session along with all associated data.
"""
from typing import Optional, Union, NamedTuple, Tuple

import os
from enum import Enum

from lns.common import config
from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.yolo.model import YoloModel
from lns.yolo.process import YoloData, YoloProcessor

class Optimizer(Enum):
    SGD: str = "sgd"
    MOMENTUM: str = "momentum"
    ADAM: str = "adam"
    RMSPROP: str = "rmsprop"

class LearningRateType(Enum):
    FIXED: str = "fixed"
    EXPONENTIAL: str = "exponential"
    COSINE_DECAY: str = "cosine_decay"
    COSINE_DECAY_RESTART: str = "cosine_decay_restart"
    PIECEWISE: str = "piecewise"

class Settings(NamedTuple):
    initial_weights: Optional[str] = None

    batch_size: int = 8
    img_size: Tuple[int, int] = (416, 416)
    letterbox_resize: bool = True

    num_epochs: int = 100
    train_evaluation_step: int = 100
    val_evaluation_epoch: int = 5
    save_epoch: int = 5

    batch_norm_decay: float = 0.99
    weight_decay: float = 5e-4
    global_step: int = 0

    num_threads: int = 10
    prefetech_buffer: int = 5

    optimizer_name: Optimizer = Optimizer.MOMENTUM
    save_optimizer: bool = True
    learning_rate_init: float = 1e-4
    lr_type: LearningRateType = LearningRateType.PIECEWISE
    lr_decay_epoch: int = 5
    lr_decay_factor: float = 0.96
    lr_lower_bound: float = 1e-6
    pw_boundaries: Tuple[int, int] = (30, 50)
    pw_values: Tuple[float, float, float] = (learning_rate_init, 3e-5, 1e-5)

    multi_scale_train: bool = True
    use_label_smooth: bool = True
    use_focal_loss: bool = True
    use_mix_up: bool = True
    use_warm_up: bool = True
    warm_up_epoch: int = 5

    nms_topk: int = 8
    nms_threshold: float = 0.25
    score_threshold: float = 0.01
    eval_threshold: float = 0.25
    use_voc_07_metric: bool = False


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

    INITIAL_WEIGHTS_NAME = "yolov3.ckpt"
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
        if "checkpoint" in checkpoints:
            checkpoints.remove("checkpoint")
        if checkpoints:
            return os.path.join(self._paths["checkpoint_folder"],
                                max(checkpoints, key=lambda checkpoint: int(checkpoint.split("_")[3])))
        return YoloTrainer.INITIAL_WEIGHTS

    def train(self, settings: Optional[Settings] = None) -> None:
        """Begin training the model."""
        if not settings:
            settings = Settings()

        # TODO: dynamically generate k-means
        self._paths["anchors_file"] = os.path.join(config.RESOURCES_ROOT, config.WEIGHTS_FOLDER_NAME, "yolo_anchors")

        from lns.yolo._lib import args

        args.train_file = self._data.get_annotations()
        args.val_file = self._data.get_annotations()
        args.restore_path = settings.initial_weights if settings.initial_weights else self._get_initial_weight()
        args.save_dir = self._paths["checkpoint_folder"] + "/"
        args.log_dir = self._paths["log_folder"]
        args.progress_log_path = self._paths["progress_file"]
        args.anchor_path = self._paths["anchors_file"]
        args.class_name_path = self._data.get_classes()

        for field, setting in zip(settings._fields, settings):
            setattr(args, field, setting)

        # args.init()

        # Importing train will begin training
        # from lns.yolo._lib import train  # pylint:disable=unused-import  # noqa

    def generate_model(self) -> Optional[YoloModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        return self.model
