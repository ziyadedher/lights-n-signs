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
        self._model = cv.SVM().load(model_path)

    def predict(self, image: np.ndarray) -> np.int64:
        """Predict the class of a given image."""
        return self._model.predict(image, dtype=np.float32)

    def eval(self, val_path: str, labels_path: str) -> None:
        """Evaluate the performance of classifier with validate data."""
        val_data = np.load(val_path, dtype=np.float32)
        val_labels = np.load(labels_path, dtype=np.int64).reshape(val_data.shape[0], 1)

        tp, fp = 0, 0
        for i in range(val_data.shape[0]):
            result = self.predict(val_data[i:])

            if result == val_labels[i]:
                tp += 1
            else:
                fp += 1

        precision = float(tp) / float(fp + tp)
        recall = float(tp) / float(val_data.shape[0])

        print("TP: {}\FP: {}\Precision: {:.2f}\Recall: {:.2f}\nF1 score: {:.2f}".format(tp, fp, precision, recall, f1_score(precision, recall)))


def f1_score(p, r):
    """Calculate the F1-Score given precision and recall.

    F1-Score is a measure combining both precision and recall, generally described as the harmonic mean of the two. 
    """
    return 0 if (p + r) == 0 else 2 * (p * r)/(p + r)