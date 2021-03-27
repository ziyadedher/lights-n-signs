"""Manages settings related to YOLO training and evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from lns.common.settings import Settings


class Optimizer(str, Enum):
    """Optimizer used for YOLO trainer."""

    SGD: str = "sgd"
    MOMENTUM: str = "momentum"
    ADAM: str = "adam"
    RMSPROP: str = "rmsprop"


class LearningRateType(str, Enum):
    """Learning rate decay type used for YOLO trainer."""

    FIXED: str = "fixed"
    EXPONENTIAL: str = "exponential"
    COSINE_DECAY: str = "cosine_decay"
    COSINE_DECAY_RESTART: str = "cosine_decay_restart"
    PIECEWISE: str = "piecewise"


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class YoloSettings(Settings):
    """Settings encapsulation for all YOLO trainer settings."""

    # Absolute path to initial weights for the model
    # If set to `None` loads the most recently trained weights for this trainer
    # or the initial weights if no trained weights exist
    initial_weights: Optional[str] = None

    # Number of images to train on per step
    batch_size: int = 32
    # Base size of the image to train on (overriden by multi_scale_train)
    img_size: Tuple[int, int] = (640, 416)
    # Whether to preserve image aspect ratio when resizing or not by using letterboxing
    letterbox_resize: bool = True

    # Number of epochs until the trainer automatically terminates
    num_epochs: int = 100
    # Number of steps between evaluating the current model on the current training batch
    train_evaluation_step: int = 100
    # Number of epochs between evaluating on the entire validation dataset
    val_evaluation_epoch: int = 2
    # Number of epochs between saving a model checkpoint
    save_epoch: int = 2

    # Percentage of data to use for validation
    val_split: float = 0.1

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
    restore_exclude: Optional[List[str]] = field(default_factory=lambda: [
        'yolov3/yolov3_head/Conv_14',
        'yolov3/yolov3_head/Conv_6',
        'yolov3/yolov3_head/Conv_22',
    ])
    # Part of the model to update
    update_part: Optional[List[str]] = field(default_factory=lambda: [
        'yolov3/yolov3_head',
    ])

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
    warm_up_epoch: int = 3

    # Number of final outputs from non-maximal suppression
    nms_topk: int = 8
    # Threshold for non-maximal suppression overlap
    nms_threshold: float = 0.01
    # Threshold for class probability in non-maximal suppresion
    score_threshold: float = 0.15
    # Thresholds for a detection to be considered correct in evaluation
    eval_threshold: float = 0.01
    # Whether or not to use 11-point VOC07 evaluation metric
    use_voc_07_metric: bool = False

    # Number of k-means clusters to compute and use
    num_clusters: int = 9
# pylint: enable=too-many-instance-attributes
