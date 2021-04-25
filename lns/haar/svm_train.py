"""SVM Classifier training script.

This script manages model training and saving.
"""
from typing import Optional, Union

import numpy as np
import cv2 as cv
import os


class SVMTrainer():
    """Manages the training environment.

    Contains and encapsulates all training setup and files under one namespace.
    """

    def __init__(self, data_path: Optional[str], labels_path: Optional[str], model_path: str) -> None:
        """Initialize a SVM trainer and save model at model_path.

        Sources data from the given <data>, if any.
        If <load> is set to False removes any existing trained model before training.
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.model_path = model_path
        self.train_data = None
        self.labels = None

    def setup(self) -> None:
        """Load datasets required for training.
        """
        if self.data_path and self.labels_path:
            self.train_data = np.load(self.data_path, allow_pickle=True)
            self.train_data = np.float32(self.train_data)

            self.labels = np.load(self.labels_path, allow_pickle=True)
            self.labels = np.int32(self.labels)
            self.labels = np.reshape(self.labels,(self.labels.shape[0], 1))

            if len(self.train_data) == 0:
                raise FileNotFoundError("Empty training data!")
        else:
            raise FileNotFoundError("Training data is not provided")

    def train(self) -> None:
        """Begin training the model.

        Train for <num_stages> stages before automatically stopping and generating the trained model.
        Train on <num_positive> positive samples and <num_negative> negative samples.
        """

        svm_model = cv.ml.SVM_create()
        svm_model.setType(cv.ml.SVM_C_SVC)
        svm_model.setKernel(cv.ml.SVM_POLY)
        svm_model.setDegree(4)
        svm_model.setGamma(1)
        svm_model.setCoef0(0)
        svm_model.setC(3.9373763856992907e-05)
        svm_model.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100000, 1e-6))
        # print(np.sum(self.labels), np.shape(self.labels))
        svm_model.train(self.train_data, cv.ml.ROW_SAMPLE, self.labels)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        svm_model.save(self.model_path+'/svm.xml')
        print(f"Saved model at {self.model_path}/svm.xml")
        # is nlt being used as the "negative" samples only 