"""SqueezeDet training script.

This script manages model training and generation.
"""
from typing import Union, Optional, List

import os
import pickle
import inspect
import easydict                     
import numpy as np     

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers

from lns.common.train import Trainer
from lns.common.dataset import Dataset
from lns.squeezedet.model import SqueezeDetModel
from lns.squeezedet.process import SqueezeDetData, SqueezeDetProcessor
from lns.squeezedet.lib import SqueezeDet

class SqueezeDetTrainer(Trainer[SqueezeDetModel, SqueezeDetData]):
    """Manages the SqueezeDet training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    _WEIGHTS_INIT_FILE = os.path.join(
        os.path.dirname(__file__), "imagenet.h5"
    )

    _squeeze: SqueezeDet
    _load: bool

    SUBPATHS = {
        "checkpoint_folder": Trainer.Subpath(
            path="checkpoint", temporal=True, required=True, path_type=Trainer.PathType.FOLDER),
        "tensorboard_folder": Trainer.Subpath(
            path="tensorboard", temporal=True, required=True, path_type=Trainer.PathType.FOLDER),
        "config_file": Trainer.Subpath(
            path="config", temporal=True, required=False, path_type=Trainer.PathType.FILE),
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
                         _processor=SqueezeDetProcessor, _method="squeezedet",
                         _load=load, _subpaths=self.SUBPATHS)

        self._squeeze = None
        self._load = (
            load and
            os.path.isdir(self.SUBPATHS["checkpoint_folder"].path) and
            len(os.listdir(self.SUBPATHS["checkpoint_folder"].path)) > 0
        )

    # @Trainer._setup
    def setup(self,) -> None:
        """Set up training the SqueezeDet model by populating the configuration.

        If <use_pretrained_weights> is set to `True` then the model will load
        pretrained weights as a fallback. If <reduce_lr_on_plateau> is set to
        `True` then the learning rate will be slowly decreased as we train if
        we hit a plateau.
        """
        self._squeeze = SqueezeDetModel(self.SUBPATHS["checkpoint_folder"].path)

    # @Trainer._train
    def train(self, epochs: int = 100, nbatches_train : int = 100) -> None:
        """Begin training the model.

        Train for <epochs> epochs before automatically stopping and
        generating the trained model.
        """
        tbCallBack = TensorBoard(log_dir=self.SUBPATHS['tensorboard_folder'].path, histogram_freq=0, write_graph=True, write_images=True)

        squeeze = self._squeeze.model
        squeeze.model.compile(
            optimizer=self.get_optimizer(),
            loss=[squeeze.loss], 
            metrics=[squeeze.loss_without_regularization, squeeze.bbox_loss, squeeze.class_loss, squeeze.conf_loss]
        )

        squeeze.model.fit_generator(
            self._data.generate_data(self._squeeze.config), 
            epochs=epochs,
            steps_per_epoch=nbatches_train, callbacks=[
                tbCallBack
            ]
        )

    def generate_model(self) -> Optional[SqueezeDetModel]:
        """Generate and return the currently available prediction model.

        Model may be `None` if there is no currently available model.
        """
        return SqueezeDetModel(checkpoint_path=self.SUBPATHS['checkpoint_folder'].path)

    def get_optimizer(self):
        opt = optimizers.Adam(lr=0.001,  clipnorm=self._squeeze.config.MAX_GRAD_NORM)
        return opt


if __name__ == '__main__':
    trainer = SqueezeDetTrainer('trainer', 'Bosch')
    trainer.setup()
    trainer.train()