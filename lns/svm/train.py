'''
SVM training script

This manages the training of the SVM and the model generation
'''
from sklearn.svm import SVC
import pickle

from lns.common.dataset import Dataset
from lns.common.train import Trainer
from lns.common.settings import SettingsType

from lns.svm.model import SVMModel
from lns.svm.process import SVMProcessor, SVMData

class SVMTrainer(Trainer[SVMModel,SVMData, SettingsType]):

    def __init__(self, name: str, dataset: Optional[Union[str, Dataset]] = None) -> None:
        """ Initialize SVM Trainer """

        super().__init__(name, dataset, _processor = SVMProcessor)

    def train(self) -> None:
        """ Train the SVM """
        self.model.fit( self.data.images, self.data.labels)


