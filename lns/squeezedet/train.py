"""SqueezeDet training script.

This script manages model training and generation.
"""
from typing import Union, Optional, List

import os
import pickle
import inspect

import easydict                                                # type: ignore
import numpy as np                                             # type: ignore
import tensorflow as tf                                        # type: ignore
import keras.backend as K                                      # type: ignore
from keras import optimizers
from keras.callbacks import (                                  # type: ignore
    Callback, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
)
import squeezedet_keras                                        # type: ignore
from squeezedet_keras.model.squeezeDet import SqueezeDet       # type: ignore
from squeezedet_keras.model.modelLoading import (              # type: ignore
    load_only_possible_weights
)
from squeezedet_keras.config import create_config              # type: ignore

from lns_common.train import Trainer
from lns_common.preprocess.preprocessing import Dataset
from lns_squeezedet.model import SqueezeDetModel
from lns_squeezedet.process import SqueezeDetData, SqueezeDetProcessor


class SqueezeDetTrainer(Trainer[SqueezeDetModel, SqueezeDetData]):
    """Manages the SqueezeDet training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    _WEIGHTS_INIT_FILE = os.path.join(
        os.path.dirname(inspect.getfile(squeezedet_keras)),
        "model", "imagenet.h5"
    )

    _config: easydict.EasyDict
    _squeeze: SqueezeDet
    _load: bool

    __subpaths = {
        "checkpoint_folder": (
            "checkpoint", True, True, "folder"
        ),
        "tensorboard_folder": (
            "tensorboard", True, True, "folder"
        ),
        "config_file": (
            "config", True, False, "file"
        )
    }

    def __init__(self, name: str,
                 dataset: Union[str, Dataset],
                 load: bool = True) -> None:
        """Initialize a SqueezeDet trainer with the given unique <name>.

        Sources data from the <dataset> given which can either be a name of
        an available dataset or a `Dataset` object. If <load> is set
        to `False` removes any existing trained checkpoints before training.
        """
        super().__init__(name, dataset,
                         _processor=SqueezeDetProcessor, _type="squeezedet",
                         _load=load, _subpaths=SqueezeDetTrainer.__subpaths)

        self._squeeze = None
        self._load = (
            load and
            os.path.isdir(self._paths["checkpoint_folder"]) and
            len(os.listdir(self._paths["checkpoint_folder"])) > 0
        )

        if self._load and os.path.isfile(self._paths["config_file"]):
            with open(self._paths["config_file"], "rb") as file:
                self._config = pickle.load(file)
            self._load_model()
            self.generate_model()
        else:
            self._config = create_config.squeezeDet_config("squeeze")

    @Trainer._setup
    def setup_squeezedet(self,) -> None:
        """Set up training the SqueezeDet model by populating the configuration.

        If <use_pretrained_weights> is set to `True` then the model will load
        pretrained weights as a fallback. If <reduce_lr_on_plateau> is set to
        `True` then the learning rate will be slowly decreased as we train if
        we hit a plateau.
        """
        raise NotImplementedError

    @Trainer._train
    def train_squeezedet(self, epochs: int = 100) -> None:
        """Begin training the model.

        Train for <epochs> epochs before automatically stopping and
        generating the trained model.
        """
        raise NotImplementedError

    def generate_model(self) -> Optional[SqueezeDetModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        raise NotImplementedError
