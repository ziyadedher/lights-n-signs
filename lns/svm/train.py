
"""SVM training script.

This manages the training of the SVM and the model generation
"""
from typing import Union
from lns.common.dataset import Dataset
from lns.common.train import Trainer

from lns.svm.model import SVMModel
from lns.svm.process import SVMProcessor, SVMData
import numpy as np


class SVMTrainer(Trainer[SVMModel, SVMData]):
    """Maintains the Training Environment for SVMs.

    Loading is not an option using sci-kit learn.
    """

    SUBPATHS: dict = {}

    def __init__(self, name: str, dataset: Union[str, Dataset], load: bool = False) -> None:
        """Initialize SVM Trainer with the given name.

        Sources data from the given <dataset>.
        """
        super().__init__(name, dataset,
                         _processor=SVMProcessor, _method=SVMProcessor.method(), _load=load,
                         _subpaths=SVMTrainer.SUBPATHS)

    def train(self) -> None:
        """Train the SVM.

        Not using any settings for now.
        """
        image_file = self._data.get_images()
        label_file = self._data.get_labels()
        self.model = SVMModel()

        images = np.load(image_file)
        labels = np.load(label_file)

        self.model.clf.fit(images, labels)
