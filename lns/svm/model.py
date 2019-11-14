'''
SVM representation

This module will contains the prediction model that will be generated by the training of the Support Vector Machine
'''

from typing import List, Tuple

from sklearn.svm import SVC
import numpy as np

from lns.common.model import Model
from lns.common.structs import Object2D, crop

class SVMModel(Model):
    """
    Class prediction using Support Vector Machine
    """

    def __init__(self) -> None:
        """
        initializing the model
        """
        self.__clf = SVC(kernel = 'linear')


    def predict( self, image: np.ndarray) -> int:
        '''Return integer as prediction of class'''
        im = image.reshape(64,64)
        return self.clf.predict(im)


