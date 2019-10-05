"""YOLOv3 trainer.

The module manages the representation of a YOLOv3 training session along with all associated data.
"""
from typing import Optional, Union, NamedTuple, Tuple, List

import os
from enum import Enum

from lns.common import config
from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.yolo.model import YoloModel
from lns.yolo.process import YoloData, YoloProcessor


class Optimizer(Enum):
    """Optimizer used for YOLO trainer."""

    SGD: str = "sgd"
    MOMENTUM: str = "momentum"
    ADAM: str = "adam"
    RMSPROP: str = "rmsprop"


class LearningRateType(Enum):
    """Learning rate decay type used for YOLO trainer."""

    FIXED: str = "fixed"
    EXPONENTIAL: str = "exponential"
    COSINE_DECAY: str = "cosine_decay"
    COSINE_DECAY_RESTART: str = "cosine_decay_restart"
    PIECEWISE: str = "piecewise"


class Settings(NamedTuple):
    """Settings encapsulation for all YOLO trainer settings."""

    # Absolute path to initial weights for the model
    # If set to `None` loads the most recently trained weights for this trainer
    # or the initial weights if no trained weights exist
    initial_weights: Optional[str] = None

    # Number of images to train on per step
    batch_size: int = 8
    # Base size of the image to train on (overriden by multi_scale_train)
    img_size: Tuple[int, int] = (416, 416)
    # Whether to preserve image aspect ratio when resizing or not by using letterboxing
    letterbox_resize: bool = True

    # Number of epochs until the trainer automatically terminates
    num_epochs: int = 100
    # Number of steps between evaluating the current model on the current training batch
    train_evaluation_step: int = 100
    # Number of epochs between evaluating on the entire validation dataset
    val_evaluation_epoch: int = 5
    # Number of epochs between saving a model checkpoint
    save_epoch: int = 5

    batch_norm_decay: float = 0.99
    weight_decay: float = 5e-4
    global_step: int = 0

    # Number of worker threads for `tf.data`
    num_threads: int = 10
    # Number of batches to prefetch
    prefetech_buffer: int = 5

    # Optimizer to use when training the network
    optimizer_name: Optimizer = Optimizer.MOMENTUM
    # Whether or not to store the optimizer in the checkpoint
    save_optimizer: bool = True
    # Initial learning rate, will be built up to in warmup and decayed after
    learning_rate_init: float = 1e-4
    lr_type: LearningRateType = LearningRateType.PIECEWISE
    # Number of epochs between decaying the learning rate
    lr_decay_epoch: int = 5
    # Exponential factor to decay learning rate by
    lr_decay_factor: float = 0.96
    # Lower bound on the learning rate
    lr_lower_bound: float = 1e-6
    # Epoch-based boundaries
    pw_boundaries: Tuple[int, int] = (30, 50)
    pw_values: Tuple[float, float, float] = (learning_rate_init, 3e-5, 1e-5)

    # Include only the following when restoring from weights
    restore_include: Optional[List[str]] = None
    # Exclude the following when restoring from weights
    restore_exclude: Optional[List[str]] = ['yolov3/yolov3_head/Conv_14', 'yolov3/yolov3_head/Conv_6',
                                            'yolov3/yolov3_head/Conv_22']
    # Part of the model to update
    update_part: Optional[List[str]] = ['yolov3/yolov3_head']

    # Whether or not to use multi scale training
    multi_scale_train: bool = True
    # Whether or not to smooth class labels
    use_label_smooth: bool = True
    # Whether or not to use focal loss
    use_focal_loss: bool = True
    # Whether or not to mix up data augmentation strategy
    use_mix_up: bool = True
    # Whether or not to start off with this number of warm-up epochs
    use_warm_up: bool = True
    # Number of epochs for warmup
    warm_up_epoch: int = 5

    # Number of final outputs from non-maximal suppression
    nms_topk: int = 8
    # Threshold for non-maximal suppression overlap
    nms_threshold: float = 0.25
    # Threshold for class probability in non-maximal suppresion
    score_threshold: float = 0.1
    # Thresholds for a detection to be considered correct in evaluation
    eval_threshold: float = 0.25
    # Whether or not to use 11-point VOC07 evaluation metric
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
        args.optimizer_name = args.optimizer_name.value
        args.lr_type = args.lr_type.value

        args.init()

        # Importing train will begin training
        from lns.yolo._lib import train  # pylint:disable=unused-import  # noqa

    def generate_model(self) -> Optional[YoloModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        return self.model
