"""SqueezeDet training script.

This script manages model training and generation.
"""
from typing import Union, Optional, List

import os
import inspect

import easydict                                                # type: ignore
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

        self._config = create_config.squeezeDet_config("squeeze")
        self._squeeze = None
        self._load = (
            load and
            os.path.isdir(self._paths["checkpoint_folder"]) and
            len(os.listdir(self._paths["checkpoint_folder"])) > 0
        )

    @Trainer._setup
    def setup_squeezedet(self, use_pretrained_weights: bool = True,
                         reduce_lr_on_plateau: bool = True) -> None:
        """Set up training the SqueezeDet model by populating the configuration.

        If <use_pretrained_weights> is set to `True` then the model will load
        pretrained weights as a fallback. If <reduce_lr_on_plateau> is set to
        `True` then the learning rate will be slowly decreased as we train if
        we hit a plateau.
        """
        self._config.init_file = (
            SqueezeDetTrainer._WEIGHTS_INIT_FILE
            if use_pretrained_weights else None
        )
        self._config.images = self._data.images
        self._config.gts = self._data.labels

        self._config.CLASS_NAMES = self.dataset.classes
        self._config.CLASSES = len(self._config.CLASS_NAMES)
        self._config.CLASS_TO_IDX = dict(zip(
            (name.lower() for name in self._config.CLASS_NAMES),
            range(self._config.CLASSES)
        ))

        self._config.STEPS = (
            len(self._config.images) // self._config.BATCH_SIZE
        )
        self._config.OPTIMIZER = "default"
        self._config.REDUCELRONPLATEAU = reduce_lr_on_plateau
        self._config.LR = self._config.LEARNING_RATE

        self._config.GPUS = 1
        self._config.CUDA_VISIBLE_DEVICES = "0"
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

        (
            self._config.ANCHOR_BOX,
            self._config.N_ANCHORS_HEIGHT,
            self._config.N_ANCHORS_WIDTH
        ) = create_config.set_anchors(self._config)
        self._config.ANCHORS = len(self._config.ANCHOR_BOX)

        K.set_session(tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True)
        ))

    @Trainer._train
    def train_squeezedet(self, epochs: int) -> None:
        """Begin training the model.

        Train for <epochs> epochs before automatically stopping and
        generating the trained model.
        """
        self._config.EPOCHS = epochs
        self._squeeze = SqueezeDet(self._config)

        # Find the checkpoint with the greatest number of trained epochs
        initial_epoch = 0
        if self._load:
            checkpoint_paths = os.listdir(self._paths["checkpoint_folder"])
            most_recent_checkpoint = checkpoint_paths[0]
            for checkpoint in checkpoint_paths:
                # Get the epoch from 'model.{epoch}-{loss}.hdf5'
                epoch = int(checkpoint.split(".")[1].split("-")[0])
                if epoch > initial_epoch:
                    initial_epoch = epoch
                    most_recent_checkpoint = checkpoint
            self._config.init_file = os.path.join(
                self._paths["checkpoint_folder"], most_recent_checkpoint
            )

        if self._config.init_file is not None:
            print(f"Resuming training from {self._config.init_file}.")
            load_only_possible_weights(
                self._squeeze.model, self._config.init_file
            )

        self._squeeze.model.compile(
            optimizer=optimizers.SGD(
                lr=self._config.LEARNING_RATE,
                decay=0,
                momentum=self._config.MOMENTUM,
                nesterov=False,
                clipnorm=self._config.MAX_GRAD_NORM
            ),
            loss=[
                self._squeeze.loss
            ],
            metrics=[
                self._squeeze.loss_without_regularization,
                self._squeeze.bbox_loss,
                self._squeeze.class_loss,
                self._squeeze.conf_loss
            ]
        )

        self._squeeze.model.fit_generator(
            self._data.generate_data(self._config),
            initial_epoch=initial_epoch,
            epochs=self._config.EPOCHS,
            steps_per_epoch=self._config.STEPS,
            callbacks=self._get_callbacks()
        )

        self.generate_model()

    def generate_model(self) -> Optional[SqueezeDetModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        if self._squeeze is not None and self._squeeze.model is not None:
            self.model = SqueezeDetModel(
                self._squeeze.model, self._config, self.dataset.classes
            )
        else:
            self.model = None
        return self.model

    def _get_callbacks(self) -> List[Callback]:
        """Get callbacks that we want for the training."""
        callbacks = []

        callbacks.append(ModelCheckpoint(
            os.path.join(
                self._paths["checkpoint_folder"],
                "model.{epoch:02d}-{loss:.2f}.hdf5"
            ),
            monitor='loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=True,
            mode='auto',
            period=1
        ))

        callbacks.append(TensorBoard(
            log_dir=self._paths["tensorboard_folder"],
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq="batch"
        ))

        if self._config.REDUCELRONPLATEAU:
            callbacks.append(ReduceLROnPlateau(
                monitor='loss',
                factor=0.1,
                verbose=1,
                patience=5,
                min_lr=0.0
            ))

        return callbacks
