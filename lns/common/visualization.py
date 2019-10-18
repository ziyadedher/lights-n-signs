"""Collection of visualization utility functions."""

from typing import Tuple, List, Optional, Union

import random

import cv2          # type: ignore
import numpy as np  # type: ignore
import argparse

from lns.common.model import Model
from lns.common.dataset import Dataset
from lns.common.structs import Object2D
from lns.common.preprocess import Preprocessor


QUIT_KEY = "q"

COLORS = np.random.uniform(0,255,size=len(labels),3))

def visualize(dataset: Union[Dataset, str], model: Optional[Model] = None, *,
              shuffle: bool = False, show_truth: bool = True) -> None:
    """Visualize the given <dataset>.

    If a <model> is supplied, visualizes with predictions from that model.
    If <shuffle> is set to `True`, randomly shuffles the images in the
    dataset before visualizing.
    """
    if isinstance(dataset, str):
        dataset = Preprocessor.preprocess(dataset)
    window_name = f"visualization_{dataset.name}"

    images = dataset.images
    if shuffle:
        random.shuffle(images)

    annotations = dataset.annotations
    for image_path in images:
        image = cv2.imread(image_path)

        if model:
            predictions = model.predict(image)
            idx = predictions.class_index
            draw_labels(image, predictions, COLORS[idx], 1)

        if show_truth:
            labels = annotations[image_path]
            idx = classes.index(labels)
            draw_labels(image, labels, (128,128,128), 2)

        cv2.imshow(window_name, image)
        if cv2.waitKey() == ord(QUIT_KEY):
            break
            '''display the label on image until quit'''

def draw_labels(image: np.ndarray, labels: List[Object2D], color: Tuple[int, int, int], thickness: int) -> None:
    """Draw the given <labels> on the given <image>."""
    for label in labels:
        cv2.rectangle(
            image,
            (label.bounds.left, label.bounds.top), (label.bounds.right, label.bounds.bottom),
            color=color, thickness=thickness
        )
        '''drawing label of class on the image'''
        cv2.putText(image,label,(label.bounds.left, label.bounds.top),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,thickness = thickness)
