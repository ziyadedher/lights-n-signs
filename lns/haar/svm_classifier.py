"""Haar model representation.

This module contains the prediction model that will be generated by the Haar
training.
"""
from typing import Optional, List, Tuple

import cv2 as cv          # type: ignore
import numpy as np  # type: ignore


class SVMClassifier():
    """A SVM classifier."""

    def __init__(self, model_path: str) -> None:
        """Initialize a SVM Classifier."""
        self._model = cv.ml.SVM_load(model_path + "/svm.xml")

    def predict(self, image: np.ndarray) -> np.int64:
        """Predict the class of a given image."""
        return self._model.predict(image)

    def eval(self, val_path: str, labels_path: str) -> None:
        """Evaluate the performance of classifier with validate data."""
        val_data = np.load(val_path, allow_pickle=True)
        val_data = np.float32(val_data).reshape((-1, 1024))
        val_labels = np.load(labels_path, allow_pickle=True)
        val_labels = np.int32(val_labels).reshape((val_labels.shape[0],1))

        predicted_results = self.predict(val_data)[1]

        tp, fp = 0, 0
        count = 0
        for i in range(val_data.shape[0]):
            if predicted_results[i] == val_labels[i]:
                tp += 1
            else:
                fp += 1
        print(count)
        precision = float(tp) / float(fp + tp)
        recall = float(tp) / float(val_data.shape[0])

        print("TP: {}\FP: {}\Precision: {:.2f}\Recall: {:.2f}\nF1 score: {:.2f}".format(tp, fp, precision, recall, f1_score(precision, recall)))


def f1_score(p, r):
    """Calculate the F1-Score given precision and recall.

    F1-Score is a measure combining both precision and recall, generally described as the harmonic mean of the two. 
    """
    return 0 if (p + r) == 0 else 2 * (p * r)/(p + r)