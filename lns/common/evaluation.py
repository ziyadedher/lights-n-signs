from typing import List, Tuple, Union

from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore

from lns.common.model import Model
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from lns.common.structs import iou


ConfusionMatrix = List[List[float]]
Metrics = List[Tuple[float, float, float]]


def confusion(model: Model, dataset: Union[str, Dataset],
              iou_threshold: float = 0.25) -> ConfusionMatrix:
    if isinstance(dataset, str):
        dataset = Preprocessor.preprocess(dataset)

    num_classes = len(dataset.classes)
    mat = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    for img_path, labels in tqdm(dataset.annotations.items()):
        preds = model.predict_path(img_path)
        label_detected = np.zeros((len(labels)), dtype=np.bool)
        pred_associated = np.zeros((len(preds)), dtype=np.bool)

        for i, label in enumerate(labels):
            for j, pred in enumerate(preds):
                if pred_associated[j]:
                    continue

                if iou(pred.bounds, label.bounds) >= iou_threshold:
                    mat[label.class_index][pred.class_index] += 1
                    if label.class_index == pred.class_index:
                        label_detected[i] = True
                        pred_associated[j] = True

        for i, label in enumerate(labels):
            if not label_detected[i]:
                mat[label.class_index][-1] += 1
        for j, pred in enumerate(preds):
            if not pred_associated[j]:
                mat[-1][pred.class_index] += 1

    return mat


def metrics(mat: ConfusionMatrix) -> Metrics:
    mets = np.empty((len(mat), 3))
    mets[:, 0] = np.diagonal(mat) / np.sum(mat, axis=1)  # precision
    mets[:, 1] = np.diagonal(mat) / np.sum(mat, axis=0)  # recall
    mets[:, 2] = 2 * (mets[:, 0] * mets[:, 1]) / (mets[:, 0] + mets[:, 1])  # f1 score
    return mets
