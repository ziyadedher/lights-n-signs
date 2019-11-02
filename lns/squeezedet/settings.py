"""Manages settings related to Squeezedet training and evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np  # type: ignore

from lns.common.settings import Settings


class Optimizer(str, Enum):
    """Optimizer used for Squeezedet trainer."""

    SGD: str = "default"
    MOMENTUM: str = "momentum"
    ADAM: str = "adam"
    RMSPROP: str = "rmsprop"


@dataclass(frozen=True)
class SqueezedetSettings(Settings):
    """Settings encapsulation for all Squeezedet trainer settings."""

    # Absolute path to initial weights for the model
    # If set to `None` loads the most recently trained weights for this trainer
    # or the initial weights if no trained weights exist
    initial_weights: Optional[str] = None

    # Whether or not to be verbose
    verbose: bool = False

    # Number of epochs to train
    num_epochs: int = 100

    # Probability to dropout a node
    dropout_ratio: float = 0.5

    # Training image width and height and number of channels
    image_width: int = 1248
    image_height: int = 384
    n_channels: int = 3

    # Number of images to train on per step
    batch_size: int = 8
    # Number of images to visualize at a time
    visualization_batch_size: int = 16

    # Optimizer to use
    optimizer: Optimizer = Optimizer.SGD

    # SGD + Momentum parameters
    weight_decay: float = 0.001
    learning_rate: float = 0.01
    max_grad_norm: float = 1.0
    momentum: float = 0.9
    # Whether or not to reduce learning rate on a plateau
    reduce_lr_on_plateau: bool = True

    # Loss function coefficients
    loss_coef_bbox: float = 5.0
    loss_coef_conf_pos: float = 75.0
    loss_coef_conf_neg: float = 100.0
    loss_coef_class: float = 1.0

    # Evaluation thresholds
    nms_thresh: float = 0.25
    prob_thresh: float = 0.0005
    top_n_detection: int = 8
    iou_threshold: float = 0.25
    final_threshold: float = 0.0

    anchor_seed: np.ndarray = field(default_factory=lambda: np.array([
        [36., 37.], [366., 174.], [115., 59.],
        [162., 87.], [38., 90.], [258., 173.],
        [224., 108.], [78., 170.], [72., 43.]
    ]))
    anchor_size: int = 16
