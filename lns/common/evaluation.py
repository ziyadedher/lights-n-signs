from typing import List, Tuple, Union, Callable, Optional

import time

from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
import cv2  # type: ignore

from lns.common import visualization
from lns.common.model import Model
from lns.common.dataset import Dataset
from lns.common.preprocess import Preprocessor
from lns.common.structs import iou


ConfusionMatrix = List[List[float]]
Metrics = List[Tuple[float, float, float]]


def _sample_annotations(dataset: Dataset, num_to_sample: Optional[int] = None) -> Dataset.Annotations:
    if num_to_sample:
        idx = np.random.choice(len(dataset.images), size=num_to_sample, replace=False)
        imgs = [dataset.images[i] for i in idx]
        return {img: dataset.annotations[img] for img in imgs}
    return dataset.annotations


def latency(model: Model, dataset: Union[str, Dataset],
            num_to_sample: Optional[int] = None) -> List[float]:
    if isinstance(dataset, str):
        dataset = Preprocessor.preprocess(dataset)

    anns = _sample_annotations(dataset, num_to_sample)
    times = np.empty(len(anns))

    for i, img_path in enumerate(tqdm(anns)):
        img = cv2.imread(img_path)
        start_time = time.time()
        model.predict(img)
        times[i] = time.time() - start_time

    return times


def confusion(model: Model, dataset: Union[str, Dataset],
              class_mapping: Callable[[int], int] = lambda x: x,
              num_to_sample: Optional[int] = None,
              iou_threshold: float = 0.25) -> Tuple[ConfusionMatrix, float]:
    if isinstance(dataset, str):
        dataset = Preprocessor.preprocess(dataset)

    num_classes = len(dataset.classes)
    mat = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    anns = _sample_annotations(dataset, num_to_sample)

    for img_path, labels in tqdm(anns.items()):
        preds = model.predict_path(img_path)
        label_detected = np.zeros(len(labels), dtype=np.bool)
        pred_associated = np.zeros(len(preds), dtype=np.bool)

        img = cv2.imread(img_path)
        visualization.draw_labels(img, labels, (255, 255, 255), 2)
        visualization.draw_labels(img, preds, (0, 0, 0), 2)
        for label in labels:
            print(label.class_index)
            print(label.bounds.top, label.bounds.left)
        for pred in preds:
            print(class_mapping(pred.class_index))
            print(pred.bounds.top, pred.bounds.left)
        cv2.imwrite(f"test.png", img)
        input()

        for i, label in enumerate(labels):
            for j, pred in enumerate(preds):
                if pred_associated[j]:
                    continue

                label_class = label.class_index
                pred_class = class_mapping(pred.class_index)

                if iou(pred.bounds, label.bounds) >= iou_threshold:
                    mat[label_class][pred_class] += 1
                    if label_class == pred_class:
                        label_detected[i] = True
                        pred_associated[j] = True

        for i, label in enumerate(labels):
            if not label_detected[i]:
                mat[label.class_index][-1] += 1
        for j, pred in enumerate(preds):
            if not pred_associated[j]:
                mat[-1][class_mapping(pred.class_index)] += 1

    return mat


def metrics(mat: ConfusionMatrix) -> Metrics:
    mets = np.empty((len(mat), 3))
    mets[:, 0] = np.diagonal(mat) / np.sum(mat, axis=1)  # precision
    mets[:, 1] = np.diagonal(mat) / np.sum(mat, axis=0)  # recall
    mets[:, 2] = 2 * (mets[:, 0] * mets[:, 1]) / (mets[:, 0] + mets[:, 1])  # f1 score
    return mets
