"""SVM Classifier training script.

This script manages model training and saving.
"""
from typing import Optional, Union

import numpy as np
import cv2 as cv

# class SVMTrainerWrapper:
#     def __init__(self, positive_folder: str, negative_folder: str, positive_np_path: str, negative_np_path: str):




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

        Create <num_samples> positive samples of the class represented by the <class_index> with given <feature_size>.
        """
        if self.data_path and self.labels_path:
            self.train_data = np.load(self.data_path, dtype=np.float32)
            self.labels = np.load(self.labels_path, dtype=np.int64).reshape(
                len(self.train_data), 1)
        else:
            raise FileNotFoundError("Training data is not provided")

    def train(self) -> None:
        """Begin training the model.

        Train for <num_stages> stages before automatically stopping and generating the trained model.
        Train on <num_positive> positive samples and <num_negative> negative samples.
        """

        svm_model = cv.ml.SVM_create()
        svm_model.setType(cv.ml.SVM_C_SVC)
        svm_model.setKernel(cv.ml.SVM_LINEAR)
        svm_model.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

        print("\n\nTraining")
        svm_model.train(samples=self.train_data,
                        layout=cv.ml.ROW_SAMPLE, responses=self.labels)
        print("\nTraining completed")

        svm_model.save(self.model_path)
        print(f"Saved model at {self.model_path}")
