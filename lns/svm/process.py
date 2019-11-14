'''
Data Processing for SVM training

Manages all data processing to generate data that is ready to be trained on by the
Open CV SVM library
'''

from typing import ClassVar, List

import os
import numpy as np
import cv2
from PIL import Image

from lns.common import config
from lns.common.structs import Object2D
from lns.common.dataset import Dataset
from lns.common.process import ProcessedData, Processor

class SVMData(ProcessedData):

    """
    Data container for the SVM processed data
    Images and Labels
    """

    __images: np.ndarray
    __labels: List[int]

    def __init__(self, images: np.ndarray, labels: List[int]) -> None:
        """ initialize the structure """
        self.__images = images
        self.__labels = labels

    @property
    def images(self) -> np.ndarray:
        """ 2D array to store images, shape = (number of images, size of image) """
        return self.__images

    @property
    def labels(self) -> np.ndarray:
        """Get array of labels"""
        return self.__labels

class SVMProcessor(Processor):
    """ Processor for processing to SVM training format """

    METHOD: ClassVar[str] = "svm"

    @classmethod
    def method(cls) -> str:
        "Return the training method"""
        return cls.METHOD

    @classmethod
    def _process(cls, dataset: Dataset) -> SVMData:
        """ No need to create new folders, just loading images """
        SVMimages = np.array([])
        SVMlabels = np.array([])

        for image_file in dataset.images:
            im = cv2.imread(image_file)
            im = cv2.resize(im, (64,64))
            im = im.flatten() #Flatten so that it can be used for SVM training
            SVMimages = np.append(SVMimages, [im], axis=0)
            SVMlabels = np.append( SVMlabels, dataset.annotations[image_file].class_index )

        return SVMData(SVMimages,SVMlabels)

